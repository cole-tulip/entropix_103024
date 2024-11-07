from typing import NamedTuple


params_1b = {
  "dim": 2048,
  "n_layers": 16,
  "n_heads": 32,
  "n_kv_heads": 8,
  "vocab_size": 128256,
  "ffn_dim_multiplier": 1.5,
  "multiple_of": 256,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": True,
  "max_seq_len": 4096
}

params_70b = {
  "dim": 8192,
  "n_layers": 80,
  "n_heads": 64,
  "n_kv_heads": 8,
  "vocab_size": 128256,
  "ffn_dim_multiplier": 1.5,
  "multiple_of": 256,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": True,
  "max_seq_len": 4096
}

class DistributedConfig(NamedTuple):
    world_size: int = 8  # Number of GPUs
    tensor_parallel_size: int = 8  # Default to using all GPUs
    activation_checkpointing: bool = True

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 and self.tensor_parallel_size > 1

class ModelParams(NamedTuple):
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool
    distributed: DistributedConfig = DistributedConfig()

    @property
    def is_parallelized(self) -> bool:
        """Helper to check if model should use tensor parallelism"""
        return self.distributed.is_distributed

# Updated parameter configs
LLAMA_1B_PARAMS = ModelParams(
    n_layers=params_1b["n_layers"],
    n_local_heads=params_1b["n_heads"],
    n_local_kv_heads=params_1b["n_kv_heads"],
    head_dim=params_1b["dim"] // params_1b["n_heads"],
    max_seq_len=params_1b["max_seq_len"],
    rope_theta=params_1b["rope_theta"],
    use_scaled_rope=params_1b["use_scaled_rope"],
    distributed=DistributedConfig(tensor_parallel_size=1, world_size=1)  # No distribution for 1B
)

LLAMA_70B_PARAMS = ModelParams(
    n_layers=params_70b["n_layers"],
    n_local_heads=params_70b["n_heads"],
    n_local_kv_heads=params_70b["n_kv_heads"],
    head_dim=params_70b["dim"] // params_70b["n_heads"],
    max_seq_len=params_70b["max_seq_len"],
    rope_theta=params_70b["rope_theta"],
    use_scaled_rope=params_70b["use_scaled_rope"]
)

def get_model_params(model_size: str = "1B") -> ModelParams:
    """Helper function to get the right model parameters"""
    if model_size == "1B":
        return LLAMA_1B_PARAMS
    elif model_size == "70B":
        return LLAMA_70B_PARAMS
    else:
        raise ValueError(f"Unknown model size: {model_size}")