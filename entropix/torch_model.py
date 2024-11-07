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

# Device selection remains unchanged
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization - remains unchanged"""
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary embeddings with proper sequence length handling"""
    input_dtype = xq.dtype
    bsz, seqlen = xq.shape[:2]
    
    # Slice freqs_cis to match our sequence length
    freqs_cis = freqs_cis[:seqlen]
    
    # Convert to float32 for numerical stability during complex operations
    reshape_xq = xq.to(dtype).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.to(dtype).reshape(*xk.shape[:-1], -1, 2)
    
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    
    xq_out = torch.stack([xq_out.real, xq_out.imag], dim=-1)
    xk_out = torch.stack([xk_out.real, xk_out.imag], dim=-1)
    
    # Convert back to original dtype for consistency with input
    xq_out = xq_out.reshape(*xq.shape[:-1], -1).to(input_dtype)
    xk_out = xk_out.reshape(*xk.shape[:-1], -1).to(input_dtype)
    
    print(f"[DEBUG] Rank {dist.get_rank() if dist.is_initialized() else 0}")
    print(f"[DEBUG] xq shape: {xq.shape}")
    print(f"[DEBUG] xk shape: {xk.shape}")
    print(f"[DEBUG] freqs_cis shape: {freqs_cis.shape}")
    print(f"[DEBUG] xq_ shape: {xq_.shape}")
    print(f"[DEBUG] freqs_cis after unsqueeze: {freqs_cis.unsqueeze(0).unsqueeze(2).shape}")
    
    return xq_out, xk_out

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

def attention(
    x: torch.Tensor, 
    layer_weights: LayerWeights, 
    model_params: ModelParams,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    process_groups: Optional[Tuple[dist.ProcessGroup, dist.ProcessGroup]] = None,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, seqlen, _ = x.shape
    input_dtype = x.dtype
    head_dim = model_params.head_dim

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    print(f"[DEBUG Rank {rank}] Starting attention:")
    print(f"[DEBUG Rank {rank}] Input shape: {x.shape}")
    print(f"[DEBUG Rank {rank}] WQ shape: {layer_weights.wq.shape}")
    print(f"[DEBUG Rank {rank}] is_parallelized: {model_params.is_parallelized}")
    print(f"[DEBUG Rank {rank}] has process_groups: {process_groups is not None}")

    # Check if we should use parallel path
    use_parallel = model_params.is_parallelized and process_groups is not None
    
    if not use_parallel:
        print(f"[DEBUG Rank {rank}] Taking non-parallel path")
        # Non-parallel path
        xq = F.linear(x, layer_weights.wq)
        xq = xq.view(bsz, seqlen, model_params.n_local_heads, head_dim)
        
        xk = F.linear(x, layer_weights.wk)
        xk = xk.view(bsz, seqlen, model_params.n_local_kv_heads, head_dim)
        
        xv = F.linear(x, layer_weights.wv)
        xv = xv.view(bsz, seqlen, model_params.n_local_kv_heads, head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        keys, values, kvcache = kvcache.update(
            xk, xv, layer_idx, cur_pos,
            model_params.n_local_heads // model_params.n_local_kv_heads
        )
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq * (head_dim ** -0.5), keys.transpose(-2, -1))
        if attn_mask is not None:
            scores = scores + attn_mask
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights.to(input_dtype)
        
        output = torch.matmul(attention_weights, values)
        output = output.transpose(1, 2)
        output = output.reshape(bsz, seqlen, -1)
        
        out = F.linear(output, layer_weights.wo)
        return out, kvcache, scores
    
    # Parallel path
    attn_group, ffn_group = process_groups
    world_size = dist.get_world_size()
    
    # Calculate local dimensions
    heads_per_gpu = model_params.n_local_heads // world_size
    local_hidden = head_dim * heads_per_gpu
    
    print(f"[DEBUG Rank {rank}] heads_per_gpu: {heads_per_gpu}")
    print(f"[DEBUG Rank {rank}] local_hidden: {local_hidden}")
    
    # Query projection with sharded weights
    xq = F.linear(x, layer_weights.wq)
    xq = xq.view(bsz, seqlen, heads_per_gpu, head_dim)
    
    # Key and value projections (replicated on all GPUs)
    xk = F.linear(x, layer_weights.wk)
    xk = xk.view(bsz, seqlen, model_params.n_local_kv_heads, head_dim)
    
    xv = F.linear(x, layer_weights.wv)
    xv = xv.view(bsz, seqlen, model_params.n_local_kv_heads, head_dim)
    
    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    
    # Update KV cache - using local kv heads
    keys, values, kvcache = kvcache.update(
        xk, xv, layer_idx, cur_pos,
        heads_per_gpu // model_params.n_local_kv_heads
    )
    
    # Reshape for attention computation
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    
    # Compute attention scores
    scores = torch.matmul(xq * (head_dim ** -0.5), keys.transpose(-2, -1))
    if attn_mask is not None:
        scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
    
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = attention_weights.to(input_dtype)
    
    output = torch.matmul(attention_weights, values)
    output = output.transpose(1, 2)
    output = output.reshape(bsz * seqlen, local_hidden)
    
    # Local projection with sharded weights
    local_out = F.linear(output, layer_weights.wo)
    
    # All-reduce to combine results from all GPUs
    dist.all_reduce(local_out, op=dist.ReduceOp.SUM, group=attn_group)
    
    return local_out.view(bsz, seqlen, -1), kvcache, scores

def feed_forward(
    x: torch.Tensor,
    layer_weights: LayerWeights,
    model_params: ModelParams,
    ffn_group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Modified feed-forward to use distributed operations"""
    if model_params.is_parallelized:
        # Split FFN computation across GPUs
        hidden = F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3)
        out = F.linear(hidden, layer_weights.w2)
        
        # All-reduce across GPUs
        if ffn_group is not None:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ffn_group)
        return out
    else:
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
    process_groups: Optional[Tuple[dist.ProcessGroup, dist.ProcessGroup]] = None,
    attn_mask: Optional[torch.Tensor]=None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    """Main transformer function with distributed support - core logic remains unchanged"""
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

        # Feed-forward network
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm)
        h_ffn = feed_forward(
            norm_x,
            xfmr_weights.layer_weights[i],
            model_params,
            process_groups[1] if process_groups is not None else None  # Pass FFN process group
        )
        h = h + h_ffn

    # Final normalization
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    
    return logits, kvcache, scores, attn_stats