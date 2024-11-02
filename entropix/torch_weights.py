from typing import List, NamedTuple, Optional
import torch
import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
from pathlib import Path

# Device selection remains unchanged
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class LayerWeights(NamedTuple):
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: List[LayerWeights]

def compare_outputs(torch_output: torch.Tensor, jax_output: jax.Array, atol: float = 1e-5, rtol: float = 1e-8) -> None:
    jax_output_np = np.array(jax_output)
    torch_output_np = torch_output.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)

    try:
        np.testing.assert_allclose(torch_output_np, jax_output_np, atol=atol, rtol=rtol)
    except AssertionError as e:
        print(f'JAX output (first 30): {jax_output_np.flatten()[:30]}')
        print(f'PyTorch output (first 30): {torch_output_np.flatten()[:30]}')
        raise e

def load_weights(
    ckpt_dir: Path = Path('weights/1B-Instruct'),
    n_layers: int = 16,
    distributed: Optional[NamedTuple] = None,
    local_rank: Optional[int] = None
) -> XfmrWeights:
    """
    Load model weights with DeepSpeed support for 70B model only.
    
    Args:
        ckpt_dir: Directory containing weight files
        n_layers: Number of transformer layers (16 for 1B, different for 70B)
        distributed: Optional distributed configuration (used for 70B)
        local_rank: Optional local rank (required for 70B with DeepSpeed)
    """
    w = {}
    layer_weights = []
    
    # Determine if this is the 70B model using DeepSpeed
    is_70b_deepspeed = (distributed is not None and 
                       distributed.use_deepspeed and 
                       n_layers > 16)  # Or another appropriate check
    
    # Device handling - default for 1B, DeepSpeed for 70B
    current_device = device
    if is_70b_deepspeed:
        if local_rank is None:
            raise ValueError("local_rank required for 70B model with DeepSpeed")
        current_device = torch.device(f"cuda:{local_rank}")
    
    with torch.inference_mode():
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
            jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            np_weight = np.array(jax_weight).astype(np.float32)
            weight = torch.from_numpy(np_weight).to(torch.bfloat16)
            
            # Apply sharding only for 70B model with DeepSpeed
            if is_70b_deepspeed and distributed.tensor_parallel_size > 1:
                if "layers." in name:
                    # Handle attention and feed-forward weights
                    if any(key in name for key in ['attention.wq', 'attention.wo', 
                                                 'feed_forward.w1', 'feed_forward.w2', 
                                                 'feed_forward.w3']):
                        shard_size = weight.size(-1) // distributed.tensor_parallel_size
                        start_idx = local_rank * shard_size
                        end_idx = start_idx + shard_size
                        weight = weight[..., start_idx:end_idx]
            
            weight = weight.to(current_device)
            compare_outputs(torch_output=weight, jax_output=jax_weight)
            w[name] = weight

        # Handle layer assignment - pipeline parallelism only for 70B with DeepSpeed
        for i in range(n_layers):
            if is_70b_deepspeed and distributed.pipeline_parallel_size > 1:
                if i % distributed.pipeline_parallel_size != local_rank:
                    continue
            
            layer_weights.append(LayerWeights(
                wq=w[f'layers.{i}.attention.wq.weight'],
                wk=w[f'layers.{i}.attention.wk.weight'],
                wv=w[f'layers.{i}.attention.wv.weight'],
                wo=w[f'layers.{i}.attention.wo.weight'],
                w1=w[f'layers.{i}.feed_forward.w1.weight'],
                w2=w[f'layers.{i}.feed_forward.w2.weight'],
                w3=w[f'layers.{i}.feed_forward.w3.weight'],
                ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
                attention_norm=w[f'layers.{i}.attention_norm.weight'],
            ))

        xfmr_weights = XfmrWeights(
            tok_embeddings=w['tok_embeddings.weight'],
            norm=w['norm.weight'],
            output=w['output.weight'],
            layer_weights=layer_weights
        )

    return xfmr_weights