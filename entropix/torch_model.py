import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Union
import logging
from entropix.distributed_types import ProcessGroups

from entropix.config import ModelParams
from entropix.torch_kvcache import KVCache
from entropix.torch_weights import SmallXfmrWeights, LargeXfmrWeights, LayerWeights
from entropix.torch_stats import AttnStats

logger = logging.getLogger(__name__)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization"""
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings."""
    dtype = xq.dtype
    seqlen = xq.size(1)
    freqs_cis = freqs_cis[:seqlen]
    
    xq_r = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.reshape(*xk.shape[:-1], -1, 2)
    
    xq_real, xq_imag = xq_r[..., 0], xq_r[..., 1]
    xk_real, xk_imag = xk_r[..., 0], xk_r[..., 1]
    
    freqs_cos = freqs_cis.real.to(dtype)
    freqs_sin = freqs_cis.imag.to(dtype)
    
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    
    out_q_real = xq_real * freqs_cos - xq_imag * freqs_sin
    out_q_imag = xq_real * freqs_sin + xq_imag * freqs_cos
    out_k_real = xk_real * freqs_cos - xk_imag * freqs_sin
    out_k_imag = xk_real * freqs_sin + xk_imag * freqs_cos
    
    xq_out = torch.stack([out_q_real, out_q_imag], dim=-1)
    xk_out = torch.stack([out_k_real, out_k_imag], dim=-1)
    
    return (
        xq_out.reshape(*xq.shape),
        xk_out.reshape(*xk.shape)
    )

def attention(
    x: torch.Tensor,
    layer_weights: LayerWeights,
    model_params: ModelParams,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    process_groups: Optional[ProcessGroups] = None,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    """Multi-head attention with GQA support."""
    bsz, seqlen, _ = x.shape
    input_dtype = x.dtype
    head_dim = model_params.head_dim
    
    use_parallel = model_params.is_parallelized and process_groups is not None
    q_heads_per_gpu = model_params.n_local_heads
    kv_heads = model_params.n_local_kv_heads
    
    # Query projection
    if use_parallel:
        xq = F.linear(x, layer_weights.wq)
        xq = xq.view(bsz, seqlen, q_heads_per_gpu, head_dim)
    else:
        xq = F.linear(x, layer_weights.wq)
        xq = xq.view(bsz, seqlen, model_params.n_local_heads, head_dim)
    
    # Key/Value projections (not split across GPUs)
    xk = F.linear(x, layer_weights.wk)
    xk = xk.view(bsz, seqlen, kv_heads, head_dim)
    
    xv = F.linear(x, layer_weights.wv)
    xv = xv.view(bsz, seqlen, kv_heads, head_dim)
    
    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    
    # Update KV cache
    keys, values, kvcache = kvcache.update(
        xk, xv, layer_idx, cur_pos,
        q_heads_per_gpu // kv_heads
    )
    
    # Reshape for attention computation
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    
    # Compute attention scores
    scaling = head_dim ** -0.5
    scores = torch.matmul(xq * scaling, keys.transpose(-2, -1))
    
    if attn_mask is not None:
        scores = scores + attn_mask
    
    attention_weights = F.softmax(scores, dim=-1).to(input_dtype)
    output = torch.matmul(attention_weights, values)
    
    # Reshape output
    output = output.transpose(1, 2)
    output = output.reshape(bsz, seqlen, -1)
    
    # Final projection with distributed handling
    if use_parallel:
        output = output.reshape(bsz * seqlen, -1)
        out = F.linear(output, layer_weights.wo)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=process_groups.attn)
        out = out.view(bsz, seqlen, -1)
    else:
        out = F.linear(output, layer_weights.wo)
    
    return out, kvcache, scores

def feed_forward(
    x: torch.Tensor,
    layer_weights: LayerWeights,
    model_params: ModelParams,
    process_groups: Optional[ProcessGroups] = None
) -> torch.Tensor:
    """Feed-forward network with distributed support."""
    if model_params.is_parallelized and process_groups is not None:
        bsz, seqlen, _ = x.shape
        x_reshaped = x.view(-1, x.size(-1))
        
        # Split computations across GPUs
        gate_proj = F.linear(x_reshaped, layer_weights.w1)
        up_proj = F.linear(x_reshaped, layer_weights.w3)
        
        hidden = F.silu(gate_proj) * up_proj
        out = F.linear(hidden, layer_weights.w2)
        
        # Combine results
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=process_groups.ffn)
        return out.view(bsz, seqlen, -1)
    else:
        hidden = F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3)
        return F.linear(hidden, layer_weights.w2)

def xfmr(
    xfmr_weights: Union[SmallXfmrWeights, LargeXfmrWeights],
    model_params: ModelParams,
    tokens: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    process_groups: Optional[ProcessGroups] = None,
    attn_mask: Optional[torch.Tensor]=None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    """Main transformer function with GQA support."""
    rank = dist.get_rank() if process_groups is not None else 0
    h = xfmr_weights.tok_embeddings[tokens]
    
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )

    # Calculate layer ranges for distributed processing
    if model_params.is_parallelized and process_groups is not None:
        layers_per_gpu = model_params.n_layers // model_params.distributed.tensor_parallel_size
        my_start_layer = rank * layers_per_gpu
        my_end_layer = my_start_layer + layers_per_gpu
    else:
        my_start_layer = 0
        my_end_layer = model_params.n_layers

    # Process layers
    for i in range(my_start_layer, my_end_layer):
        local_idx = i - my_start_layer
        layer_weights = xfmr_weights.layer_weights[local_idx]
        
        # Attention block
        norm_x = rms_norm(h, layer_weights.attention_norm)
        h_attn, kvcache, scores = attention(
            norm_x,
            layer_weights,
            model_params,
            cur_pos,
            i,
            freqs_cis,
            kvcache,
            process_groups,
            attn_mask
        )
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn

        # Feed-forward block
        norm_x = rms_norm(h, layer_weights.ffn_norm)
        h_ffn = feed_forward(
            norm_x,
            layer_weights,
            model_params,
            process_groups
        )
        h = h + h_ffn

        # Synchronize between GPUs if needed
        if model_params.is_parallelized and process_groups is not None:
            dist.all_reduce(h, op=dist.ReduceOp.SUM, group=process_groups.ffn)

    # Final processing
    normalized = rms_norm(h, xfmr_weights.norm)
    
    if isinstance(xfmr_weights, LargeXfmrWeights):
        logits = F.linear(normalized, xfmr_weights.output)
        if process_groups is not None:
            dist.all_reduce(logits, op=dist.ReduceOp.SUM, group=process_groups.ffn)
    else:
        logits = normalized
    
    return logits, kvcache, scores, attn_stats