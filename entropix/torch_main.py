from typing import NamedTuple, Optional, Tuple
import os
import torch
import torch.distributed as dist
import deepspeed
import accelerate
import tyro
import math
from pathlib import Path
from functools import partial

from entropix.config import get_model_params
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import load_weights
from entropix.torch_sampler import sample
from entropix.prompts import create_prompts_from_csv, prompt

# Device selection logic preserved
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """Preserved scaling function"""
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192

    # Original scaling logic preserved
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

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Preserved frequency computation"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)
    freqs = freqs.unsqueeze(0)
    freqs = t * freqs
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
    """Preserved attention mask building"""
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
    return mask

def setup_distributed(model_params):
    """Initialize distributed training when needed"""
    if model_params.distributed.use_deepspeed:
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=True,
            init_method="env://"
        )
    else:
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

def main(
    model_size: str = "1B",
    prompts_file: Optional[str] = None,
    weights_dir: Optional[Path] = None,
    local_rank: int = 0,  # Added this
):
    model_params = get_model_params(model_size)
    
    # Original distributed setup logic preserved
    if model_size == "70B":
        setup_distributed(model_params)  # Keep this existing function
    
    # Original device selection logic preserved
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    with torch.inference_mode():
        xfmr_weights = load_weights(
            ckpt_dir=weights_dir,
            n_layers=model_params.n_layers,
            distributed=model_params.distributed if model_size == "70B" else None,
            local_rank=local_rank if model_size == "70B" else None  # Add this parameter
        )

        tokenizer = Tokenizer('entropix/tokenizer.model')
        
        def generate(xfmr_weights, model_params, tokens):
            """Preserved generate function"""
            # ... rest of generate function remains exactly the same ...

        # Handle prompts exactly as before
        if prompts_file:
            prompts = create_prompts_from_csv(prompts_file)
            for i, prompt_text in enumerate(prompts):
                print(f"\nProcessing prompt {i+1}/{len(prompts)}")
                print("Prompt:", prompt_text[:100], "...")
                tokens = tokenizer.encode(prompt_text, bos=True, eos=False, allowed_special='all')
                generate(xfmr_weights, model_params, tokens)
        else:
            # Use default prompt
            tokens = tokenizer.encode(prompt, bos=True, eos=False, allowed_special='all')
            generate(xfmr_weights, model_params, tokens)
            
if __name__ == '__main__':
    tyro.cli(main)