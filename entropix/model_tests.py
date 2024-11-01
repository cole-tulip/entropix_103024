import time
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from entropix.config import LLAMA_1B_PARAMS
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import load_weights

@dataclass
class TestResult:
    name: str
    success: bool
    duration: float
    memory_peak: float
    details: Dict

class ModelTester:
    def __init__(self):
        from entropix.config import LLAMA_1B_PARAMS
        from entropix.tokenizer import Tokenizer
        from entropix.torch_kvcache import KVCache
        from entropix.torch_model import xfmr
        from entropix.torch_weights import load_weights

        self.model_params = LLAMA_1B_PARAMS
        self.tokenizer = Tokenizer('entropix/tokenizer.model')
        self.xfmr_weights = load_weights()
        self.device = torch.device('cuda')
        self.results: List[TestResult] = []

    def _measure_memory(self) -> float:
        """Return peak GPU memory usage in GB"""
        return torch.cuda.max_memory_allocated() / 1024**3

    def _clear_memory(self):
        """Clear GPU cache and reset memory stats"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def run_basic_inference(self, prompt: str) -> TestResult:
        start_time = time.time()
        self._clear_memory()
        
        try:
            # Tokenize input
            tokens = torch.tensor([self.tokenizer.encode(prompt, bos=True, eos=False)], 
                                dtype=torch.long).to(self.device)
            
            with torch.inference_mode():
                bsz, seqlen = tokens.shape
                kvcache = KVCache.new(
                    self.model_params.n_layers,
                    bsz,
                    self.model_params.max_seq_len,
                    self.model_params.n_local_kv_heads,
                    self.model_params.head_dim
                ).to(self.device)
                
                freqs_cis = torch.ones((self.model_params.max_seq_len, 
                                      self.model_params.head_dim // 2), 
                                     dtype=torch.float32).to(self.device)
                
                logits, _, _, _ = xfmr(
                    self.xfmr_weights,
                    self.model_params,
                    tokens,
                    0,
                    freqs_cis[:seqlen],
                    kvcache
                )
                
                next_token = torch.argmax(logits[:, -1], dim=-1)
                result = self.tokenizer.decode([next_token.item()])

            success = len(result) > 0
            details = {
                "output_token": result,
                "input_length": seqlen,
                "memory_allocated": torch.cuda.memory_allocated() / 1024**3
            }
            
        except Exception as e:
            success = False
            details = {"error": str(e)}

        return TestResult(
            name="basic_inference",
            success=success,
            duration=time.time() - start_time,
            memory_peak=self._measure_memory(),
            details=details
        )

    def run_memory_test(self, max_length: int = 512) -> TestResult:
        start_time = time.time()
        self._clear_memory()
        
        try:
            # Generate a long sequence gradually
            prompt = "This is a test of memory usage with increasing sequence length."
            base_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            memory_profile = []
            
            for length in range(100, max_length, 100):
                # Repeat tokens to desired length
                tokens = base_tokens * (length // len(base_tokens) + 1)
                tokens = tokens[:length]
                tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                with torch.inference_mode():
                    bsz, seqlen = tokens.shape
                    kvcache = KVCache.new(
                        self.model_params.n_layers,
                        bsz,
                        self.model_params.max_seq_len,
                        self.model_params.n_local_kv_heads,
                        self.model_params.head_dim
                    ).to(self.device)
                    
                    freqs_cis = torch.ones((self.model_params.max_seq_len, 
                                          self.model_params.head_dim // 2), 
                                         dtype=torch.float32).to(self.device)
                    
                    _ = xfmr(
                        self.xfmr_weights,
                        self.model_params,
                        tokens,
                        0,
                        freqs_cis[:seqlen],
                        kvcache
                    )
                    
                    memory_profile.append((length, torch.cuda.memory_allocated() / 1024**3))
                
                self._clear_memory()

            success = True
            details = {
                "memory_profile": memory_profile,
                "max_memory": max(m for _, m in memory_profile)
            }
            
        except Exception as e:
            success = False
            details = {"error": str(e)}

        return TestResult(
            name="memory_test",
            success=success,
            duration=time.time() - start_time,
            memory_peak=self._measure_memory(),
            details=details
        )

    def run_speed_test(self, num_iterations: int = 10) -> TestResult:
        start_time = time.time()
        self._clear_memory()
        
        try:
            prompt = "Testing inference speed."
            tokens = torch.tensor([self.tokenizer.encode(prompt, bos=True, eos=False)], 
                                dtype=torch.long).to(self.device)
            
            inference_times = []
            
            with torch.inference_mode():
                for _ in range(num_iterations):
                    iter_start = time.time()
                    
                    bsz, seqlen = tokens.shape
                    kvcache = KVCache.new(
                        self.model_params.n_layers,
                        bsz,
                        self.model_params.max_seq_len,
                        self.model_params.n_local_kv_heads,
                        self.model_params.head_dim
                    ).to(self.device)
                    
                    freqs_cis = torch.ones((self.model_params.max_seq_len, 
                                          self.model_params.head_dim // 2), 
                                         dtype=torch.float32).to(self.device)
                    
                    _ = xfmr(
                        self.xfmr_weights,
                        self.model_params,
                        tokens,
                        0,
                        freqs_cis[:seqlen],
                        kvcache
                    )
                    
                    inference_times.append(time.time() - iter_start)
                    
                    # Clear memory between iterations
                    self._clear_memory()

            success = True
            details = {
                "mean_inference_time": np.mean(inference_times),
                "std_inference_time": np.std(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times)
            }
            
        except Exception as e:
            success = False
            details = {"error": str(e)}

        return TestResult(
            name="speed_test",
            success=success,
            duration=time.time() - start_time,
            memory_peak=self._measure_memory(),
            details=details
        )

    def run_all_tests(self) -> List[TestResult]:
        """Run all tests and return results"""
        tests = [
            (self.run_basic_inference, ("Hello, world!",)),
            (self.run_memory_test, (512,)),
            (self.run_speed_test, (10,))
        ]
        
        results = []
        for test_func, args in tests:
            try:
                result = test_func(*args)
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    name=test_func.__name__,
                    success=False,
                    duration=0,
                    memory_peak=0,
                    details={"error": str(e)}
                ))
            finally:
                self._clear_memory()
        
        return results

def print_results(results: List[TestResult]):
    """Print test results in a readable format"""
    print("\n=== Model Test Results ===\n")
    
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.name}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Peak Memory: {result.memory_peak:.2f}GB")
        
        if isinstance(result.details, dict):
            for key, value in result.details.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        print()

if __name__ == "__main__":
    tester = ModelTester()
    results = tester.run_all_tests()
    print_results(results)