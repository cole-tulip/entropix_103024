import os
import time
import torch
from pathlib import Path
from typing import Dict

def run_verification() -> Dict[str, bool]:
    results = {}
    start_time = time.time()
    
    print("Starting quick verification...")
    
    try:
        # 1. Verify weights directory
        weights_path = Path('weights/1B-Instruct')
        results['weights_exist'] = weights_path.exists() and any(weights_path.glob('*.npy'))
        print(f"✓ Weights check: {results['weights_exist']}")
        
        # 2. Import all required modules
        from entropix.config import LLAMA_1B_PARAMS
        from entropix.tokenizer import Tokenizer
        from entropix.torch_kvcache import KVCache
        from entropix.torch_model import xfmr
        from entropix.torch_weights import load_weights
        results['imports_successful'] = True
        print("✓ Imports successful")
        
        # 3. Load tokenizer
        tokenizer = Tokenizer('entropix/tokenizer.model')
        test_tokens = tokenizer.encode("Hello, world!", bos=True, eos=False)
        results['tokenizer_works'] = len(test_tokens) > 0
        print(f"✓ Tokenizer check: {len(test_tokens)} tokens generated")
        
        # 4. Load model weights
        model_params = LLAMA_1B_PARAMS
        xfmr_weights = load_weights()
        results['weights_loaded'] = True
        print("✓ Weights loaded successfully")
        
        # 5. Quick inference test
        test_input = "Complete this sentence: The quick brown fox"
        tokens = torch.tensor([tokenizer.encode(test_input, bos=True, eos=False)], dtype=torch.long).cuda()
        
        with torch.inference_mode():
            # Create cache and run inference
            bsz, seqlen = tokens.shape
            freqs_cis = torch.ones((model_params.max_seq_len, model_params.head_dim // 2), dtype=torch.float32).cuda()
            kvcache = KVCache.new(
                model_params.n_layers,
                bsz,
                model_params.max_seq_len,
                model_params.n_local_kv_heads,
                model_params.head_dim
            ).cuda()
            
            logits, _, _, _ = xfmr(
                xfmr_weights,
                model_params,
                tokens,
                0,  # cur_pos
                freqs_cis[:seqlen],
                kvcache
            )
            
            next_token = torch.argmax(logits[:, -1], dim=-1)
            result = tokenizer.decode([next_token.item()])
        
        results['inference_works'] = len(result) > 0
        print(f"✓ Inference check: Generated token: {result}")
        
        # 6. Memory check
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        results['memory_reasonable'] = memory_allocated < 16  # Less than 16GB for 1B model
        print(f"✓ Memory check: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        results['error'] = str(e)
        
    duration = time.time() - start_time
    print(f"\nVerification completed in {duration:.2f} seconds")
    
    return results

if __name__ == "__main__":
    results = run_verification()
    
    print("\nSummary:")
    all_passed = all(v is True for v in results.values() if isinstance(v, bool))
    if all_passed:
        print("✅ All checks passed")
    else:
        print("❌ Some checks failed:")
        for k, v in results.items():
            if isinstance(v, bool) and not v:
                print(f"  - {k}: Failed")
            elif not isinstance(v, bool):
                print(f"  - {k}: {v}")