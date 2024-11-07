import numpy as np
import torch
import torch.distributed as dist
from typing import List, NamedTuple, Optional
from pathlib import Path
import logging
import torch.distributed as dist

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

class XfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: List[LayerWeights]

def convert_to_torch_bfloat16(np_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch bfloat16 tensor via uint16."""
    # Convert through uint16 as intermediate step
    return torch.from_numpy(np_array.view(np.uint16)).view(torch.bfloat16)

def load_weights(
    ckpt_dir: Path = Path('weights/1B-Instruct'),
    n_layers: int = 16,
    distributed: Optional[NamedTuple] = None,
    local_rank: Optional[int] = None,
    device: Optional[torch.device] = None
) -> XfmrWeights:
    w = {}
    layer_weights = []
    
    is_distributed = distributed is not None and distributed.is_distributed
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
        
    logger.info(f"Rank {local_rank if is_distributed else 0} starting weight loading")
    
    with torch.inference_mode():
        if is_distributed:
            world_size = distributed.tensor_parallel_size
            layers_per_gpu = n_layers // world_size
            my_start_layer = local_rank * layers_per_gpu
            my_end_layer = my_start_layer + layers_per_gpu
            hidden_per_gpu = 8192 // world_size  # 1024 per GPU for 8-way parallel

            # Validate dimensions
            assert hidden_per_gpu * world_size == 8192, "Hidden dimension must divide evenly"
            assert n_layers % world_size == 0, "Layers must divide evenly across GPUs"
            
            for file in sorted(ckpt_dir.glob("*.npy")):
                name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
                
                # Load shared weights on all ranks
                if "tok_embeddings.weight" in name or "norm.weight" in name or "output.weight" in name:
                    weight_np = np.load(str(file))
                    w[name] = convert_to_torch_bfloat16(weight_np).to(device)
                    
                elif "layers." in name:
                    layer_num = int(name.split('.')[1])
                    if my_start_layer <= layer_num < my_end_layer:
                        if "attention.wq.weight" in name:
                            # Shard query projections by output dimension
                            weight_np = np.load(str(file))
                            start_idx = local_rank * hidden_per_gpu
                            end_idx = (local_rank + 1) * hidden_per_gpu
                            weight_np = weight_np[start_idx:end_idx, :]
                            w[name] = convert_to_torch_bfloat16(weight_np).to(device)
                            
                        elif "attention.wo.weight" in name:
                            # Shard output projections by input dimension
                            weight_np = np.load(str(file))
                            start_idx = local_rank * hidden_per_gpu  
                            end_idx = (local_rank + 1) * hidden_per_gpu
                            weight_np = weight_np[:, start_idx:end_idx]
                            w[name] = convert_to_torch_bfloat16(weight_np).to(device)
                            
                        elif "attention.wk.weight" in name or "attention.wv.weight" in name:
                            # Load full KV weights on all GPUs
                            weight_np = np.load(str(file))
                            w[name] = convert_to_torch_bfloat16(weight_np).to(device)
                            
                        else:
                            # FFN weights - shard by output dimension
                            weight_np = np.load(str(file))
                            if "feed_forward.w1" in name or "feed_forward.w3" in name:
                                start_idx = local_rank * (weight_np.shape[0] // world_size)
                                end_idx = (local_rank + 1) * (weight_np.shape[0] // world_size)
                                weight_np = weight_np[start_idx:end_idx, :]
                            w[name] = convert_to_torch_bfloat16(weight_np).to(device)

            # Build layer weights for this rank's layers
            for i in range(my_start_layer, my_end_layer):
                layer_weights.append(LayerWeights(
                    wq=w[f'layers.{i}.attention.wq.weight'],
                    wk=w[f'layers.{i}.attention.wk.weight'],
                    wv=w[f'layers.{i}.attention.wv.weight'], 
                    wo=w[f'layers.{i}.attention.wo.weight'],
                    w1=w[f'layers.{i}.feed_forward.w1.weight'],
                    w2=w[f'layers.{i}.feed_forward.w2.weight'],
                    w3=w[f'layers.{i}.feed_forward.w3.weight'],
                    attention_norm=w[f'layers.{i}.attention_norm.weight'],
                    ffn_norm=w[f'layers.{i}.ffn_norm.weight']
                ))

            return XfmrWeights(
                tok_embeddings=w['tok_embeddings.weight'],
                norm=w['norm.weight'],
                output=w['output.weight'],
                layer_weights=layer_weights
            )