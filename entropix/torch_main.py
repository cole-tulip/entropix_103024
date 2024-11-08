import os
import torch
import torch.distributed as dist
import logging
import tyro
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from entropix.prompts import create_prompts_from_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from entropix.config import get_model_params
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.distributed_types import ProcessGroups
from entropix.torch_weights import load_weights
from entropix.torch_sampler import sample

# Essential CUDA/NCCL settings
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

class DistributedSetup:
    """Simplified distributed setup handler."""
    
    def __init__(self, world_size: int = 8):
        self.required_world_size = world_size
        self.initialized = False
        self._validate_environment()
        
    def _validate_environment(self):
        """Validate essential environment variables."""
        required_vars = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        
        if missing_vars:
            raise RuntimeError(
                f"Missing required environment variables: {missing_vars}. "
                "Are you using torchrun?"
            )
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        
        if self.world_size != self.required_world_size:
            raise RuntimeError(
                f"Expected world size of {self.required_world_size}, "
                f"got {self.world_size}"
            )

    def _setup_device(self):
        """Set up GPU device for this process."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for distributed setup")
            
        gpu_count = torch.cuda.device_count()
        if gpu_count != self.required_world_size:
            raise RuntimeError(
                f"Expected {self.required_world_size} GPUs, found {gpu_count}"
            )
            
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

    def initialize(self) -> Tuple[int, torch.device, ProcessGroups]:
        """Initialize distributed environment."""
        try:
            self._setup_device()
            
            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method="env://"
            )
            
            # Create single process group for tensor parallelism
            process_group = dist.new_group(
                ranks=list(range(self.world_size)),
                backend="nccl"
            )
            
            # Create ProcessGroups with same group for both operations
            process_groups = ProcessGroups(attn=process_group, ffn=process_group)
            
            if self.local_rank == 0:
                logger.info(f"Distributed setup complete with {self.world_size} GPUs")
                
            self.initialized = True
            return self.local_rank, self.device, process_groups
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed setup: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False

def setup_inference(model_params) -> Tuple[int, torch.device, Optional[ProcessGroups]]:
    """Set up inference environment."""
    if not model_params.distributed.is_distributed:
        return 0, torch.device("cuda"), None
        
    try:
        dist_setup = DistributedSetup(world_size=model_params.distributed.world_size)
        return dist_setup.initialize()
    except Exception as e:
        logger.error(f"Failed to setup inference: {e}")
        raise

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """Position embedding scaling."""
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

def precompute_freqs_cis(
    dim: int, 
    end: int, 
    theta: float = 500000.0, 
    use_scaled: bool = False, 
    device: torch.device = None
) -> torch.Tensor:
    """Precompute position embeddings."""
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
        if use_scaled:
            freqs = apply_scaling(freqs)

        t = torch.arange(end, dtype=torch.float32, device=device).unsqueeze(1)
        freqs = freqs.unsqueeze(0)
        freqs = t * freqs
        return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int, device: torch.device) -> torch.Tensor:
    """Build attention mask."""
    mask = None
    if seqlen > 1:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask])
    return mask

def generate(xfmr_weights, model_params, tokens, process_groups=None):
    """Generate text using the loaded model with MCTS search."""        
    device = xfmr_weights.tok_embeddings.device
    tokens = torch.tensor([tokens], device=device)
    
    # Initialize KV cache
    kv_cache = KVCache.new(
        layers=model_params.n_layers,
        bsz=tokens.shape[0],
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

    cur_pos = 0
    attn_mask = build_attn_mask(seqlen=tokens.shape[1], start_pos=0, device=device)

    # Initial forward pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits, kv_cache, attention_scores, attn_stats = xfmr(
            xfmr_weights=xfmr_weights,
            model_params=model_params,
            tokens=tokens,
            cur_pos=cur_pos,
            freqs_cis=freqs_cis,
            kvcache=kv_cache,
            process_groups=process_groups,
            attn_mask=attn_mask
        )

        # Sample next token
        next_token = sample(tokens, logits, attention_scores)

    # Sync token distribution in distributed mode
    if model_params.is_parallelized and process_groups is not None:
        dist.broadcast(next_token, src=0, group=process_groups.attn)

    tokens = torch.cat([tokens, next_token], dim=1)
    return tokens

@dataclass
class Args:
    """Command line arguments"""
    model_size: str = "1B"
    weights_dir: Optional[Path] = None
    prompts_file: Optional[Path] = None

def main(
    model_size: str = "1B",
    prompts_file: Optional[str] = None,
    weights_dir: Optional[Path] = None,
):
    """Main execution function."""
    model_params = get_model_params(model_size)
    dist_setup = None
    
    try:
        # Set up distributed environment if needed
        local_rank, device, process_groups = setup_inference(model_params)
        
        if model_params.distributed.is_distributed:
            logger.info(f"Running in distributed mode on GPU {local_rank}")
        
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Load weights
            xfmr_weights = load_weights(
                ckpt_dir=weights_dir or Path(f'weights/{model_size}-Instruct'),
                n_layers=model_params.n_layers,
                distributed=model_params.distributed if model_size == "70B" else None,
                local_rank=local_rank if model_size == "70B" else None,
                device=device
            )

            tokenizer = Tokenizer('entropix/tokenizer.model')
            
            # Use one of the predefined prompts from prompts.py
            from entropix.prompts import prompt2  # Using the Spain capital prompt as example
            input_text = prompt2
            logger.info(f"Using prompt: {input_text[:200]}...")  # Log first 200 chars

            input_tokens = tokenizer.encode(
                input_text,
                bos=True,
                eos=False,
                allowed_special="all"  # Important for the special tokens in prompt
            )

            generated_tokens = generate(
                xfmr_weights, 
                model_params, 
                input_tokens, 
                process_groups
            )
                
            if local_rank == 0:
                print(tokenizer.decode(generated_tokens[0].tolist()))
                
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        # Clean up
        if model_params.distributed.is_distributed and dist_setup:
            dist_setup.cleanup()

if __name__ == "__main__":
    tyro.cli(main)