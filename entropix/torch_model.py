import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from entropix.config import ModelParams
from entropix.torch_kvcache import KVCache
from entropix.torch_weights import XfmrWeights, LayerWeights
from entropix.torch_stats import AttnStats

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization - no changes needed"""
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary embeddings - no changes needed"""
    input_dtype = xq.dtype
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(input_dtype), xk_out.to(input_dtype)

def attention(
    x: torch.Tensor, 
    layer_weights: LayerWeights, 
    model_params: ModelParams,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    """Modified attention to handle distributed computation for 70B model"""
    bsz, seqlen, _ = x.shape
    input_dtype = x.dtype

    # Calculate repeats for grouped-query attention
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    
    # For 70B model with tensor parallelism
    if model_params.distributed.tensor_parallel_size > 1:
        local_rank = dist.get_rank()
        tp_size = model_params.distributed.tensor_parallel_size
        head_dim = model_params.head_dim
        
        # Split attention heads across GPUs
        heads_per_gpu = model_params.n_local_heads // tp_size
        start_head = local_rank * heads_per_gpu
        end_head = start_head + heads_per_gpu
        
        # Slice weights for this GPU's portion
        local_wq = layer_weights.wq[start_head * head_dim : end_head * head_dim, :]
        local_wk = layer_weights.wk
        local_wv = layer_weights.wv
        local_wo = layer_weights.wo[:, start_head * head_dim : end_head * head_dim]
        
        # Project queries, keys, and values with local weights
        xq = F.linear(x, local_wq).reshape(bsz, seqlen, heads_per_gpu, head_dim)
        xk = F.linear(x, local_wk).reshape(bsz, seqlen, model_params.n_local_kv_heads, head_dim)
        xv = F.linear(x, local_wv).reshape(bsz, seqlen, model_params.n_local_kv_heads, head_dim)
    else:
        # Original logic for 1B model
        xq = F.linear(x, layer_weights.wq).reshape(bsz, seqlen, model_params.n_local_heads, model_params.head_dim)
        xk = F.linear(x, layer_weights.wk).reshape(bsz, seqlen, model_params.n_local_kv_heads, model_params.head_dim)
        xv = F.linear(x, layer_weights.wv).reshape(bsz, seqlen, model_params.n_local_kv_heads, model_params.head_dim)

    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # Update KV cache and get full keys/values
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

    # Reshape for attention computation
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2).transpose(2, 3)
    values = values.transpose(1, 2)

    # Compute attention scores
    scores = torch.matmul(xq, keys) / math.sqrt(model_params.head_dim)
    scores = scores.to(torch.float32)

    if cur_pos == 0 and attn_mask is not None:
        scores = scores + attn_mask

    # Apply masking
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_scores = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    
    # Compute attention weights
    attention_weights = F.softmax(padded_scores, dim=-1)
    attention_weights = attention_weights.to(input_dtype)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, values)
    
    # Reshape output
    output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
    
    # Final projection with appropriate weights
    if model_params.distributed.tensor_parallel_size > 1:
        # Each GPU computes its portion of the output
        out = F.linear(output, local_wo)
        # All-reduce across GPUs to get final output
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    else:
        out = F.linear(output, layer_weights.wo)
    
    return out, kvcache, scores

def feed_forward(
    x: torch.Tensor,
    layer_weights: LayerWeights,
    model_params: ModelParams
) -> torch.Tensor:
    """Modified feed-forward to handle distributed computation"""
    if model_params.distributed.tensor_parallel_size > 1:
        local_rank = dist.get_rank()
        tp_size = model_params.distributed.tensor_parallel_size
        
        # Split FFN computation across GPUs
        ffn_dim = layer_weights.w1.size(0)
        chunk_size = ffn_dim // tp_size
        start_idx = local_rank * chunk_size
        end_idx = start_idx + chunk_size
        
        # Get local portions of weights
        local_w1 = layer_weights.w1[start_idx:end_idx, :]
        local_w2 = layer_weights.w2[:, start_idx:end_idx]
        local_w3 = layer_weights.w3[start_idx:end_idx, :]
        
        # Compute local portion of FFN
        hidden = F.silu(F.linear(x, local_w1)) * F.linear(x, local_w3)
        out = F.linear(hidden, local_w2)
        
        # All-reduce across GPUs
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
    else:
        # Original logic for 1B model
        return F.linear(
            F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3),
            layer_weights.w2
        )

def xfmr(
    xfmr_weights: XfmrWeights,
    model_params: ModelParams,
    tokens: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor]=None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    """Main transformer function with distributed support"""
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )

    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(
            norm_x,
            xfmr_weights.layer_weights[i],
            model_params,
            cur_pos,
            i,
            freqs_cis,
            kvcache,
            attn_mask=attn_mask
        )
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn
        h = h + feed_forward(
            rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm),
            xfmr_weights.layer_weights[i],
            model_params
        )

    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats