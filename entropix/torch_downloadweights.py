import os
import torch
import jax.numpy as jnp
import ml_dtypes
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import huggingface_hub

# Constants
MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct'
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

def reverse_permute(tensor: torch.Tensor, n_heads: int = 32, dim1: int = 4096, dim2: int = 4096) -> torch.Tensor:
    """Reverse permute the attention weights."""
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for HuggingFace import issue."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def download_weights(model_id: str = MODEL_ID, out_dir: Path = Path('weights/1B-Instruct')):
    """Download and convert weights from HuggingFace to the required format."""
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from {model_id}...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            offload_folder="/tmp/offload",
            token=TOKEN
        )
        
        print("Converting weights...")
        with torch.no_grad():
            state_dict = hf_model.state_dict()
            for hf_name, param in state_dict.items():
                print(f' {hf_name}: {param.shape=}')
                name = translate_key(hf_name)
                if name.endswith('wq.weight'):
                    param = reverse_permute(param, n_heads=32, dim1=2048, dim2=2048)  # 1B
                    #param = reverse_permute(param, n_heads=24, dim1=3072, dim2=3072)  # 3B
                    #param = reverse_permute(param, n_heads=32, dim1=4096, dim2=4096)  # 7B
                    #param = reverse_permute(param, n_heads=64, dim1=8192, dim2=8192)   # 70B
                    #param = reverse_permute(param, n_heads=96, dim1=12288, dim2=12288)   # 123B
                    #param = reverse_permute(param, n_heads=128, dim1=16384, dim2=16384) # 405B
                    #param = reverse_permute(param, n_heads=128, dim1=12288, dim2=12288) # DSV2
                    #param = reverse_permute(param, n_heads=64, dim1=12288, dim2=12288) # Commandr+
                    #param = reverse_permute(param, n_heads=48, dim1=6144, dim2=6144)    # Mixtral8x22B
                elif name.endswith('wk.weight'): #wk.weight
                    param = reverse_permute(param, n_heads=8, dim1=512, dim2=2048)  # 1B
                    #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=3072)  # 3B
                    #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=4096)  # 7B
                    #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=8192)   # 70B
                    #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=12288)   # 123B
                    #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=16384)  # 405B
                    #param = reverse_permute(param, n_heads=128, dim1=12288, dim2=12288)  # DSV2
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=12288)  # Commandr+
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=6144)    # Mixtral8x22B
                else:
                    pass
                bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                save_path = out_dir / f'{name}.npy'
                print(f'Writing {hf_name} as {name} to {save_path}')
                jnp.save(str(save_path), bf16_out)

    print(f"Successfully downloaded and converted weights to {out_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--token", type=str, help="HuggingFace token")
    parser.add_argument("--out_dir", type=str, default="weights/1B-Instruct", help="Output directory")
    args = parser.parse_args()
    
    if args.token:
        TOKEN = args.token
    
    download_weights(args.model_id, Path(args.out_dir))