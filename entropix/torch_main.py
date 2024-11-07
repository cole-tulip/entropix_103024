from typing import NamedTuple, Optional, Tuple, List
import os
import torch
import torch.distributed as dist
import logging
import tyro
import math
import datetime
from pathlib import Path
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from entropix.config import get_model_params
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import load_weights
from entropix.torch_sampler import sample
from entropix.prompts import create_prompts_from_csv, prompt

def setup_inference(model_params):
    """Setup distributed inference with proper process groups"""
    print(f"[Setup] distributed config: {model_params.distributed}")
    print(f"[Setup] is_distributed: {model_params.distributed.is_distributed}")
    
    if not model_params.distributed.is_distributed:
        print("[Setup] Taking non-distributed path")
        return 0, torch.device("cuda"), None, None
        
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Setup] local_rank: {local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize process groups
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    print(f"[Setup] world_size: {world_size}")
    
    if world_size != model_params.distributed.world_size:
        print(f"World size mismatch: {world_size} vs {model_params.distributed.world_size}")
        cleanup_distributed()
        raise ValueError("World size mismatch")
    
    # Create process groups
    ranks = list(range(world_size))
    attn_group = dist.new_group(ranks=ranks)
    ffn_group = dist.new_group(ranks=ranks)
    print(f"[Setup] Created process groups")
    
    return local_rank, device, attn_group, ffn_group

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_device_and_setup():
    """Unified device selection logic"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    torch.set_float32_matmul_precision('high')
    return device

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """Preserved scaling function - no changes needed"""
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,
                scaled
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32, device: torch.device = None) -> torch.Tensor:
    """Preserved frequency computation with device parameter"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)
    freqs = freqs.unsqueeze(0)
    freqs = t * freqs
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int, device: torch.device) -> torch.Tensor:
    """Preserved attention mask building with device parameter"""
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask])
    return mask

def main(
    model_size: str = "1B",
    prompts_file: Optional[str] = None,
    weights_dir: Optional[Path] = None,
):
    """Main execution function"""
    model_params = get_model_params(model_size)
    local_rank, device, attn_group, ffn_group = setup_inference(model_params)
    
    try:
        with torch.inference_mode():
            xfmr_weights = load_weights(
                ckpt_dir=weights_dir,
                n_layers=model_params.n_layers,
                distributed=model_params.distributed if model_size == "70B" else None,
                local_rank=local_rank if model_size == "70B" else None,
                device=device
            )

            tokenizer = Tokenizer('entropix/tokenizer.model')
            
            def generate(xfmr_weights, model_params, tokens: List[int], process_groups=None):
                tokens = torch.tensor([tokens], device=device)
                bsz = tokens.shape[0]
                seqlen = tokens.shape[1]
                
                # Initialize KV cache
                kv_cache = KVCache.new(
                    layers=model_params.n_layers,
                    bsz=bsz,
                    max_seq_len=model_params.max_seq_len,
                    kv_heads=model_params.n_local_kv_heads,
                    head_dim=model_params.head_dim, 
                    device=device
                )

                # Precompute position embeddings
                freqs_cis = precompute_freqs_cis(
                    dim=model_params.head_dim,
                    end=model_params.max_seq_len,
                    theta=model_params.rope_theta,
                    use_scaled=model_params.use_scaled_rope,
                    device=device
                )

                # Build attention mask
                attn_mask = build_attn_mask(seqlen=seqlen, start_pos=0, device=device)

                # Main generation loop / forward pass
                logits, kv_cache, attention_scores, attn_stats = xfmr(
                    xfmr_weights=xfmr_weights,
                    model_params=model_params,
                    tokens=tokens,
                    cur_pos=0,
                    freqs_cis=freqs_cis,
                    kvcache=kv_cache,
                    process_groups=process_groups,  # Pass process_groups directly
                    attn_mask=attn_mask
                )
                
                next_token = sample(
                    gen_tokens=tokens,
                    logits=logits,
                    attention_scores=attention_scores
                )
                
                tokens = torch.cat([tokens, next_token], dim=1)
                print(tokenizer.decode(tokens[0].tolist()))
                return tokens
    finally:
        # Clean up distributed resources
        if model_params.distributed.is_distributed:
            cleanup_distributed()

if __name__ == '__main__':
    tyro.cli(main)