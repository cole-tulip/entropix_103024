import os
from dotenv import load_dotenv
import torch
import jax.numpy as jnp
import ml_dtypes
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import huggingface_hub
from typing import Optional

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Model configurations
MODEL_CONFIGS = {
    '1b': {
        'model_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'n_heads': 32,
        'n_kv_heads': 8,
        'dim': 2048,
        'out_dir': 'weights/1B-Instruct'
    },
    '70b': {
        'model_id': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
        'n_heads': 64,
        'n_kv_heads': 8,
        'dim': 8192,
        'out_dir': 'weights/70B-Instruct'
    }
}

TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not TOKEN:
    raise ValueError('HUGGINGFACE_TOKEN environment variable is not set.')

def translate_key(in_key: str):
    """Translate HuggingFace model keys to expected format."""
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key.endswith('down_proj'):
            out_key = out_key.replace('down_proj', 'w2')
        elif out_key.endswith('gate_proj'):
            out_key = out_key.replace('gate_proj', 'w1')
        elif out_key.endswith('up_proj'):
            out_key = out_key.replace('up_proj', 'w3')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
        else:
            print(f"Don't know how to handle {in_key=}")
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
    return f'{out_key}.weight'

def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int, is_key: bool = False, n_kv_heads: int = None) -> torch.Tensor:
    """Reverse permute the attention weights."""
    if is_key:
        if n_kv_heads is None:
            raise ValueError("n_kv_heads must be provided for key weights")
        return tensor
    else:
        return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for HuggingFace import issue."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def download_weights(model_size: str = '1b'):
    """Download and convert weights from HuggingFace to the required format."""
    if model_size.lower() not in MODEL_CONFIGS:
        raise ValueError(f"Model size {model_size} not supported. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size.lower()]
    out_dir = Path(config['out_dir'])
    model_id = config['model_id']
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model {model_id} to {out_dir}...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        # For 70B model, use device_map='auto' to handle memory
        device_map = 'auto' if model_size.lower() == '70b' else None
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            offload_folder="/tmp/offload",
            device_map=device_map,
            token=TOKEN
        )
        
        print("Converting weights...")
    with torch.no_grad():
        state_dict = hf_model.state_dict()
        for hf_name, param in state_dict.items():
            print(f' {hf_name}: {param.shape=}')
            name = translate_key(hf_name)
            
            # Handle both key and value weights for GQA
            if model_size.lower() == '70b' and ('wk.weight' in name or 'wv.weight' in name):
                print(f"Processing {'key' if 'wk' in name else 'value'} weight with shape {param.shape}")
                param = reverse_permute(
                    param, 
                    n_heads=config['n_heads'],
                    dim1=config['dim'],
                    dim2=config['dim'],
                    is_key=True,  # Treat both k and v weights the same way for GQA
                    n_kv_heads=config['n_kv_heads']
                )
                print(f"After permute: {param.shape}")
            elif name.endswith('wq.weight'):
                param = reverse_permute(
                    param, 
                    n_heads=config['n_heads'],
                    dim1=config['dim'],
                    dim2=config['dim']
                )
            
            if name.endswith('wq.weight'):
                param = reverse_permute(
                    param, 
                    n_heads=config['n_heads'],
                    dim1=config['dim'],
                    dim2=config['dim']
                )
            elif name.endswith('wk.weight'):
                print(f"Processing key weight with shape {param.shape}")
                param = reverse_permute(
                    param, 
                    n_heads=config['n_heads'],
                    dim1=config['dim'],
                    dim2=config['dim'],
                    is_key=True,
                    n_kv_heads=config['n_kv_heads']
                )
                print(f"After permute: {param.shape}")
                
            # Move parameter to CPU if it's on GPU
            param = param.cpu() if param.device.type != 'cpu' else param
                
            bf16_np_out = param.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
            bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
            save_path = out_dir / f'{name}.npy'
            print(f'Writing {hf_name} as {name} to {save_path}')
            jnp.save(str(save_path), bf16_out)

    print(f"Successfully downloaded and converted weights to {out_dir}")

if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass
    
    @dataclass
    class Args:
        model_size: str = "1B"  # Match the main.py convention
        weights_dir: Optional[Path] = None
    
    args = tyro.cli(Args)
    
    # Convert model size to lowercase for dict lookup
    model_size = args.model_size.lower()
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Model size {args.model_size} not supported. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    # If weights_dir is provided, override the default
    if args.weights_dir:
        MODEL_CONFIGS[model_size]['out_dir'] = str(args.weights_dir)
    
    download_weights(model_size=model_size)