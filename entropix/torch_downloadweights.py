import os
import sys
import psutil
import gc
import logging
import math
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.distributed as dist
import jax.numpy as jnp
import numpy as np
import ml_dtypes
from dotenv import load_dotenv
import huggingface_hub
import safetensors
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import tyro
import warnings
warnings.filterwarnings('ignore', message='A module that was compiled using NumPy 1.x')

# Configure logging with timestamps and both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'model_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Model configurations with memory requirements and architecture details
MODEL_CONFIGS = {
    '1b': {
        'model_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'n_heads': 32,
        'n_kv_heads': 8,
        'dim': 2048,
        'n_layers': 16,
        'out_dir': 'weights/1B-Instruct',
        'required_disk_gb': 5,
        'required_ram_gb': 8,
        'estimated_gpu_gb': 4,
        'use_device_map': False,  # Can load fully into memory
        'head_dim': 64  # 2048 // 32
    },
    '70b': {
        'model_id': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
        'n_heads': 64,
        'n_kv_heads': 8,
        'dim': 8192,
        'n_layers': 80,
        'layers_per_gpu': 10,  # 80 layers / 8 GPUs
        'hidden_per_gpu': 1024,  # 8192 / 8 GPUs
        'out_dir': 'weights/70B-Instruct',
        'required_disk_gb': 200,
        'required_ram_gb': 50,
        'estimated_gpu_gb': 80,
        'use_device_map': True,  # Needs memory management
        'head_dim': 128  # 8192 // 64
    }
}

# Constants
GB = 1024 * 1024 * 1024  # 1 GB in bytes

def check_system_requirements(config: dict):
    """Verify system has enough resources"""
    logger.info("Checking system requirements...")
    
    # Check available disk space
    disk = psutil.disk_usage(os.path.abspath('.'))
    required_space = config['required_disk_gb'] * GB
    if disk.free < required_space:
        raise RuntimeError(
            f"Not enough disk space. Need {config['required_disk_gb']}GB, "
            f"have {disk.free / GB:.2f}GB"
        )
    logger.info(f"Disk space check passed: {disk.free / GB:.2f}GB available")
    
    # Check available RAM
    ram = psutil.virtual_memory()
    required_ram = config['required_ram_gb'] * GB
    if ram.available < required_ram:
        raise RuntimeError(
            f"Not enough available RAM. Need {config['required_ram_gb']}GB, "
            f"have {ram.available / GB:.2f}GB"
        )
    logger.info(f"RAM check passed: {ram.available / GB:.2f}GB available")
    
    # Check GPU if needed
    if torch.cuda.is_available() and config.get('use_device_map', False):
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < config['estimated_gpu_gb'] * GB:
            logger.warning(
                f"GPU memory might be insufficient. Have {gpu_memory / GB:.2f}GB, "
                f"recommended {config['estimated_gpu_gb']}GB"
            )
        else:
            logger.info(f"GPU memory check passed: {gpu_memory / GB:.2f}GB available")

def verify_token_and_model(model_id: str):
    """Verify HuggingFace token and model access"""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise ValueError('HUGGINGFACE_TOKEN environment variable is not set')
    
    try:
        model_info = huggingface_hub.model_info(model_id)
        logger.info(f"Successfully verified access to model: {model_id}")
        
        # Safer way to calculate size
        sizes = [
            sibling.size 
            for sibling in model_info.siblings 
            if hasattr(sibling, 'size') and sibling.size is not None
        ]
        
        if sizes:
            total_size = sum(sizes)
            logger.info(f"Model size (from siblings): {total_size / GB:.2f}GB")
        else:
            logger.warning("Could not determine model size from API")
            
        return token
    except Exception as e:
        logger.error(f"Error accessing model info: {str(e)}")
        logger.warning("Continuing without model size information...")
        return token

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for HuggingFace import issue with flash attention"""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def log_memory_usage(stage: str):
    """Log current memory usage with timestamps"""
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 / 1024  # MB
    memory_info = f"{stage} - RAM: {ram_usage:.2f}MB"
    
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        memory_info += f", GPU Used: {gpu_usage:.2f}MB, GPU Cached: {gpu_cached:.2f}MB"
    
    logger.info(memory_info)

def clear_memory():
    """Thoroughly clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int, head_dim: int, 
                   is_key: bool = False, n_kv_heads: Optional[int] = None) -> torch.Tensor:
    """
    Reverse permute attention weights for GQA architecture.
    
    Args:
        tensor: Input tensor to permute
        n_heads: Number of attention heads
        dim1: First dimension (usually model dimension)
        dim2: Second dimension (usually model dimension)
        head_dim: Dimension of each head
        is_key: Whether this is a key/value projection
        n_kv_heads: Number of key/value heads (required if is_key=True)
    """
    logger.debug(f"Permuting tensor with shape {tensor.shape}")
    logger.debug(f"Parameters: n_heads={n_heads}, dim1={dim1}, dim2={dim2}, head_dim={head_dim}")
    
    if is_key:
        if n_kv_heads is None:
            raise ValueError("n_kv_heads must be provided for key/value weights")
        logger.debug(f"Processing key/value weight with {n_kv_heads} KV heads")
        # For GQA, reshape to group key/value heads
        tensor = tensor.view(n_kv_heads, n_heads // n_kv_heads, head_dim, dim2)
        tensor = tensor.transpose(0, 1).reshape(dim1, dim2)
    else:
        logger.debug("Processing query weight")
        # For query weights, handle the interleaved structure
        tensor = tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        tensor = tensor.transpose(1, 2).reshape(dim1, dim2)
    
    logger.debug(f"Output tensor shape: {tensor.shape}")
    return tensor

def translate_key(in_key: str) -> str:
    """Translate HuggingFace model keys to expected format."""
    # First remove any '.weight' suffix and 'model.' prefix
    out_key = in_key.replace('model.', '').replace('.weight', '')
    
    # Map attention layer names
    if 'self_attn.q_proj' in out_key:
        out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
    elif 'self_attn.k_proj' in out_key:
        out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
    elif 'self_attn.v_proj' in out_key:
        out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
    elif 'self_attn.o_proj' in out_key:
        out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
    
    # Map MLP layer names
    elif 'mlp.gate_proj' in out_key:
        out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
    elif 'mlp.up_proj' in out_key:
        out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
    elif 'mlp.down_proj' in out_key:
        out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
    
    # Map norm layer names
    elif 'input_layernorm' in out_key:
        out_key = out_key.replace('input_layernorm', 'attention_norm')
    elif 'post_attention_layernorm' in out_key:
        out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        
    # Add base weight mappings
    elif out_key == 'embed_tokens':
        out_key = 'tok_embeddings'
    elif out_key == 'norm':
        out_key = 'norm'
    elif out_key == 'lm_head':
        out_key = 'output'
        
    # Add weight suffix back
    return f'{out_key}.weight'

def process_param(param: torch.Tensor, config: dict, name: str) -> torch.Tensor:
    """
    Process a parameter tensor based on its name and configuration.
    
    Args:
        param: The parameter tensor to process
        config: Model configuration dictionary
        name: Name of the parameter
    """
    logger.debug(f"Processing parameter {name} with shape {param.shape}")
    
    # Move to CPU for processing
    param = param.cpu()
    
    if 'attention.wq' in name:
        logger.info(f"Processing query weight with {config['n_heads']} heads")
        param = reverse_permute(
            param,
            n_heads=config['n_heads'],
            dim1=config['dim'],
            dim2=config['dim'],
            head_dim=config['head_dim'],
            is_key=False
        )
    elif 'attention.wk' in name or 'attention.wv' in name:
        logger.info(f"Processing key/value weight with {config['n_kv_heads']} KV heads")
        param = reverse_permute(
            param,
            n_heads=config['n_heads'],
            dim1=config['dim'],
            dim2=config['dim'],
            head_dim=config['head_dim'],
            is_key=True,
            n_kv_heads=config['n_kv_heads']
        )
    
    return param

def save_weight(param: torch.Tensor, save_path: Path):
   """Save a weight tensor using appropriate format based on path"""
   try:
       if save_path.exists():
           logger.info(f"File {save_path} already exists, skipping...")
           return
           
       # Ensure we're working with bfloat16 and on CPU
       if param.dtype != torch.bfloat16:
           param = param.to(torch.bfloat16)
       param = param.detach().cpu()

       # Always use .weight extension
       save_path = save_path.with_suffix('.weight')
       weights_dict = {"weight": param}
       save_file(weights_dict, str(save_path))
       
       logger.debug(f"Successfully saved to {save_path}")
   except Exception as e:
       logger.error(f"Failed to save weight to {save_path}: {e}")
       raise

def load_saved_weight(path: Path) -> torch.Tensor:
    """Load a saved weight tensor from safetensors format"""
    with safe_open(str(path.with_suffix('.safetensors')), framework="pt", device="cpu") as f:
        return f.get_tensor("weight")

def check_dependencies():
    """Verify and configure dependencies"""
    logger.info("Checking dependencies...")
    
    # Configure PyTorch
    if torch.cuda.is_available():
        # Set to high precision for more accurate computation
        torch.set_float32_matmul_precision('high')
        logger.info("CUDA available, using high precision float32 matmul")
    
    # Log versions
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"ML-dtypes version: {ml_dtypes.__version__}")
    
    # Verify NumPy 2.x
    numpy_version = tuple(map(int, np.__version__.split('.')))
    if numpy_version[0] < 2:
        raise RuntimeError("NumPy 2.x is required for ml-dtypes compatibility")
        
    # Verify ml_dtypes has bfloat16
    if not hasattr(ml_dtypes, 'bfloat16'):
        raise RuntimeError("ml_dtypes.bfloat16 not available")

def download_1b_model(config: dict, out_dir: Path, token: str):
    """Download and process 1B model weights in one go"""
    logger.info("Downloading 1B model weights...")
    log_memory_usage("Before model download")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            config['model_id'],
            torch_dtype=torch.bfloat16,
            token=token
        )
        
        logger.info("Processing weights...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                try:
                    logger.info(f"Processing {name}")
                    out_name = translate_key(name)
                    processed_param = process_param(param, config, name)
                    save_path = out_dir / out_name
                    save_weight(processed_param, save_path)
                    
                    if param.numel() * param.element_size() > 1 * GB:
                        clear_memory()
                        log_memory_usage(f"After processing {name}")
                except Exception as e:
                    logger.error(f"Failed processing {name}: {e}")
                    raise
    
    clear_memory()
    log_memory_usage("After model processing")
    logger.info("1B model download completed successfully")

def download_70b_model(config: dict, out_dir: Path, token: str, start_layer: int, end_layer: int):
   """Download and process 70B model weights in chunks using safetensors"""
   logger.info(f"Processing 70B model layers {start_layer} to {end_layer}...")
   log_memory_usage("Before layer download")
   
   out_dir.mkdir(parents=True, exist_ok=True)
   layer_dir = out_dir / 'layers'
   layer_dir.mkdir(parents=True, exist_ok=True)

   safetensor_dir = Path(huggingface_hub.snapshot_download(
       config['model_id'],
       token=token,
       allow_patterns=["*.safetensors"],
       ignore_patterns=["*.msgpack", "*.h5"],
   ))
   
   shard_files = sorted(safetensor_dir.glob("*.safetensors"))
   logger.info(f"Found {len(shard_files)} safetensor shards")

   try:
       # Phase 1: Process base weights in root dir
       base_weight_patterns = ['embed_tokens', 'norm', 'lm_head']
       logger.info("Processing base weights...")
       for shard_file in shard_files:
           with safe_open(shard_file, framework="pt", device="cpu") as f:
               for key in f.keys():
                   # Only process non-layer weights
                   if not 'layers.' in key:
                       for pattern in base_weight_patterns:
                           if pattern in key:
                               try:
                                   translated_name = translate_key(key)
                                   weight_path = out_dir / translated_name
                                   
                                   if weight_path.with_suffix('.weight').exists():
                                       logger.info(f"Base weight {translated_name} already exists, skipping...")
                                       continue
                                       
                                   logger.info(f"Processing base weight {key} -> {translated_name}")
                                   tensor = f.get_tensor(key)
                                   processed_param = process_param(tensor.to(torch.bfloat16), config, key)
                                   save_weight(processed_param, weight_path)
                               except Exception as e:
                                   logger.error(f"Failed processing base weight {key}: {e}")
                                   raise

       # Phase 2: Process layer weights in layer subdirectories
       for layer_idx in range(start_layer, end_layer):
           layer_path = layer_dir / f'layer_{layer_idx}'
           layer_path.mkdir(exist_ok=True)
           
           # Define expected weight names for each layer
           layer_weights = {
               f'layers.{layer_idx}.attention.wq.weight',
               f'layers.{layer_idx}.attention.wk.weight',
               f'layers.{layer_idx}.attention.wv.weight',
               f'layers.{layer_idx}.attention.wo.weight',
               f'layers.{layer_idx}.feed_forward.w1.weight',
               f'layers.{layer_idx}.feed_forward.w2.weight',
               f'layers.{layer_idx}.feed_forward.w3.weight',
               f'layers.{layer_idx}.attention_norm.weight',
               f'layers.{layer_idx}.ffn_norm.weight'
           }
           
           # Check for existing weights
           existing_weights = {f.stem for f in layer_path.glob("*.weight")}
           missing_weights = {w.replace('.weight', '') for w in layer_weights} - existing_weights
           if not missing_weights:
               logger.info(f"Layer {layer_idx} already fully processed, skipping...")
               continue
           
           logger.info(f"Processing layer {layer_idx}")
           
           for shard_file in shard_files:
               with safe_open(shard_file, framework="pt", device="cpu") as f:
                   for key in f.keys():
                       if f'layers.{layer_idx}.' in key:
                           try:
                               translated_name = translate_key(key)
                               weight_path = layer_path / translated_name
                               
                               if weight_path.with_suffix('.weight').exists():
                                   logger.info(f"Weight {translated_name} already exists, skipping...")
                                   continue
                                   
                               logger.info(f"Processing {key} -> {translated_name}")
                               tensor = f.get_tensor(key)
                               processed_param = process_param(tensor.to(torch.bfloat16), config, key)
                               save_weight(processed_param, weight_path)
                           except Exception as e:
                               logger.error(f"Failed processing {key} in layer {layer_idx}: {e}")
                               raise
               
           clear_memory()
           log_memory_usage(f"After layer {layer_idx}")

   except Exception as e:
       logger.error(f"Error during model processing: {e}")
       raise
   finally:
       clear_memory()

@dataclass
class Args:
    """Command line arguments"""
    model_size: str = "1B"
    weights_dir: Optional[Path] = None
    start_layer: int = 0
    end_layer: Optional[int] = None

def main(args: Args):
    """Main execution function with proper dependency checking"""
    try:
        check_dependencies()
        
        # Rest of the main function remains the same
        model_size = args.model_size.lower()
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Model size {args.model_size} not supported. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_size]
        out_dir = args.weights_dir if args.weights_dir else Path(config['out_dir'])
        
        check_system_requirements(config)
        token = verify_token_and_model(config['model_id'])
        
        try:
            if model_size == '1b':
                download_1b_model(config, out_dir, token)
            else:  # 70b
                if args.end_layer is None:
                    args.end_layer = args.start_layer + config['layers_per_gpu']
                
                if not (0 <= args.start_layer < args.end_layer <= config['n_layers']):
                    raise ValueError(
                        f"Invalid layer range: {args.start_layer} to {args.end_layer}. "
                        f"Must be within 0 to {config['n_layers']}"
                    )
                
                download_70b_model(config, out_dir, token, args.start_layer, args.end_layer)
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
        finally:
            clear_memory()
            
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)