import numpy as np
import torch
import torch.distributed as dist
from typing import List, NamedTuple, Optional, Union
from pathlib import Path
import logging
from safetensors import safe_open
from entropix.distributed_types import ProcessGroups

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class SmallXfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    layer_weights: List[LayerWeights]

class LargeXfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: List[LayerWeights]

def load_weight(path: Path) -> torch.Tensor:
    """Load a single weight file saved with save_weight function."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor("weight")

def load_weights(
    ckpt_dir: Path = Path('weights/1B-Instruct'),
    n_layers: int = 16,
    distributed: Optional[NamedTuple] = None,
    local_rank: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Union[SmallXfmrWeights, LargeXfmrWeights]:
    """
    Load and distribute model weights with GQA support.
    For 70B model: 10 layers per GPU, 8 KV heads, 64 query heads distributed across GPUs.
    """
    w = {}
    layer_weights = []
    
    is_distributed = distributed is not None and distributed.is_distributed
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
        
    logger.info(f"Rank {local_rank if is_distributed else 0} starting weight loading")
    
    with torch.inference_mode():
        # Load non-layer weights from root directory first
        for file in ['tok_embeddings.weight', 'norm.weight', 'output.weight']:
            if (ckpt_dir / file).exists():
                tensor = load_weight(ckpt_dir / file)
                w[file.replace('.weight', '')] = tensor.to(torch.bfloat16).to(device)

        if not is_distributed:
            # Non-distributed path unchanged
            for i in range(n_layers):
                layer_dir = ckpt_dir / 'layers' / f'layer_{i}'
                layer_files = {
                    'attention.wq': f'layers.{i}.attention.wq.weight',
                    'attention.wk': f'layers.{i}.attention.wk.weight',
                    'attention.wv': f'layers.{i}.attention.wv.weight',
                    'attention.wo': f'layers.{i}.attention.wo.weight',
                    'feed_forward.w1': f'layers.{i}.feed_forward.w1.weight',
                    'feed_forward.w2': f'layers.{i}.feed_forward.w2.weight',
                    'feed_forward.w3': f'layers.{i}.feed_forward.w3.weight',
                    'attention_norm': f'layers.{i}.attention_norm.weight',
                    'ffn_norm': f'layers.{i}.ffn_norm.weight'
                }
                
                for weight_name, file_name in layer_files.items():
                    key = f'layers.{i}.{weight_name}'
                    tensor = load_weight(layer_dir / file_name)
                    w[key] = tensor.to(torch.bfloat16).to(device)

                layer_weights.append(LayerWeights(
                    wq=w[f'layers.{i}.attention.wq'],
                    wk=w[f'layers.{i}.attention.wk'],
                    wv=w[f'layers.{i}.attention.wv'],
                    wo=w[f'layers.{i}.attention.wo'],
                    w1=w[f'layers.{i}.feed_forward.w1'],
                    w2=w[f'layers.{i}.feed_forward.w2'],
                    w3=w[f'layers.{i}.feed_forward.w3'],
                    attention_norm=w[f'layers.{i}.attention_norm'],
                    ffn_norm=w[f'layers.{i}.ffn_norm']
                ))
            
            return SmallXfmrWeights(
                tok_embeddings=w['tok_embeddings'],
                norm=w['norm'],
                output=w['output'],
                layer_weights=layer_weights
            )
            
        else:
            # Distributed path for 70B
            world_size = distributed.tensor_parallel_size
            layers_per_gpu = n_layers // world_size
            my_start_layer = local_rank * layers_per_gpu
            my_end_layer = my_start_layer + layers_per_gpu

            logger.info(f"Rank {local_rank}: Loading layers {my_start_layer}-{my_end_layer-1}")
            
            # GQA-specific dimensions
            q_heads_per_gpu = 64 // world_size
            hidden_per_gpu = 8192 // world_size
            kv_heads = 8
            
            # Validate dimensions for GQA
            assert hidden_per_gpu * world_size == 8192, "Hidden dimension must divide evenly"
            assert n_layers % world_size == 0, "Layers must divide evenly for 10-per-GPU split"
            assert 64 % world_size == 0, "Query heads must divide evenly across GPUs"
            
            # Load this GPU's assigned layers
            for i in range(my_start_layer, my_end_layer):
                layer_dir = ckpt_dir / 'layers' / f'layer_{i}'
                
                if not layer_dir.exists():
                    raise RuntimeError(f"Missing layer directory: {layer_dir}")
                
                # Calculate local layer index for this GPU's layer_weights list
                local_layer_idx = i - my_start_layer
                logger.info(f"Rank {local_rank}: Loading global layer {i} as local index {local_layer_idx}")

                # Handle query projection with GQA pattern
                tensor_path = layer_dir / f'layers.{i}.attention.wq.weight'
                if not tensor_path.exists():
                    raise RuntimeError(f"Rank {local_rank}: Missing weight file: {tensor_path}")
                    
                tensor = load_weight(tensor_path)
                start_idx = local_rank * hidden_per_gpu
                end_idx = (local_rank + 1) * hidden_per_gpu
                w[f'layers.{local_layer_idx}.attention.wq'] = tensor[start_idx:end_idx, :].to(torch.bfloat16).to(device)
                
                # Load full KV weights (not sharded)
                w[f'layers.{local_layer_idx}.attention.wk'] = load_weight(
                    layer_dir / f'layers.{i}.attention.wk.weight'
                ).to(torch.bfloat16).to(device)
                
                w[f'layers.{local_layer_idx}.attention.wv'] = load_weight(
                    layer_dir / f'layers.{i}.attention.wv.weight'
                ).to(torch.bfloat16).to(device)
                
                # Handle output projection
                tensor = load_weight(layer_dir / f'layers.{i}.attention.wo.weight')
                w[f'layers.{local_layer_idx}.attention.wo'] = tensor[:, start_idx:end_idx].to(torch.bfloat16).to(device)
                
                # Load FFN weights with proper sharding
                for ffn_name in ['w1', 'w2', 'w3']:
                    tensor = load_weight(layer_dir / f'layers.{i}.feed_forward.{ffn_name}.weight')
                    if ffn_name in ['w1', 'w3']:
                        # For w1 and w3, shard the output dimension
                        start_idx = local_rank * (tensor.shape[0] // world_size)
                        end_idx = (local_rank + 1) * (tensor.shape[0] // world_size)
                        tensor = tensor[start_idx:end_idx, :]
                    elif ffn_name == 'w2':
                        # For w2, shard the input dimension
                        start_idx = local_rank * (tensor.shape[1] // world_size)
                        end_idx = (local_rank + 1) * (tensor.shape[1] // world_size)
                        tensor = tensor[:, start_idx:end_idx]
                    w[f'layers.{local_layer_idx}.feed_forward.{ffn_name}'] = tensor.to(torch.bfloat16).to(device)
                
                # Load norm weights (not sharded)
                w[f'layers.{local_layer_idx}.attention_norm'] = load_weight(
                    layer_dir / f'layers.{i}.attention_norm.weight'
                ).to(torch.bfloat16).to(device)
                
                w[f'layers.{local_layer_idx}.ffn_norm'] = load_weight(
                    layer_dir / f'layers.{i}.ffn_norm.weight'
                ).to(torch.bfloat16).to(device)
                
                layer_weights.append(LayerWeights(
                    wq=w[f'layers.{local_layer_idx}.attention.wq'],
                    wk=w[f'layers.{local_layer_idx}.attention.wk'],
                    wv=w[f'layers.{local_layer_idx}.attention.wv'],
                    wo=w[f'layers.{local_layer_idx}.attention.wo'],
                    w1=w[f'layers.{local_layer_idx}.feed_forward.w1'],
                    w2=w[f'layers.{local_layer_idx}.feed_forward.w2'],
                    w3=w[f'layers.{local_layer_idx}.feed_forward.w3'],
                    attention_norm=w[f'layers.{local_layer_idx}.attention_norm'],
                    ffn_norm=w[f'layers.{local_layer_idx}.ffn_norm']
                ))
                
                logger.debug(f"Rank {local_rank} loaded layer {i} as local index {local_layer_idx}")
                
            # Debug check layer indices
            for i, layer in enumerate(layer_weights):
                if not isinstance(layer, LayerWeights):
                    logger.error(f"Rank {local_rank}: Layer {i} is type {type(layer)}")
                if not hasattr(layer, 'attention_norm'):
                    logger.error(f"Rank {local_rank}: Layer {i} missing attention_norm")

            return LargeXfmrWeights(
                tok_embeddings=w['tok_embeddings'],
                norm=w['norm'],
                output=w['output'],
                layer_weights=layer_weights
            )