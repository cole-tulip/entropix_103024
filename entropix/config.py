from typing import NamedTuple

# Base model parameters from original config
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
    "n_kv_heads": 8,  # GQA: 8 KV heads shared across 64 query heads
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
    layers_per_gpu: int = 10  # 80 layers / 8 GPUs for 70B
    hidden_per_gpu: int = 1024  # 8192 / 8 GPUs for 70B
    activation_checkpointing: bool = True

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 and self.tensor_parallel_size > 1

class ModelParams(NamedTuple):
    dim: int  # Model dimension
    n_layers: int  # Total number of layers
    n_heads: int  # Number of query heads
    n_kv_heads: int  # Number of key/value heads (8 for GQA)
    head_dim: int  # Dimension per head
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool
    distributed: DistributedConfig = DistributedConfig()

    @property
    def is_parallelized(self) -> bool:
        return self.distributed.is_distributed
        
    @property
    def n_local_heads(self) -> int:
        """Number of query heads per GPU when distributed"""
        if self.is_parallelized:
            return self.n_heads // self.distributed.tensor_parallel_size
        return self.n_heads
        
    @property
    def n_local_kv_heads(self) -> int:
        """KV heads are not split across GPUs in GQA"""
        return self.n_kv_heads

# Updated parameter configs
LLAMA_1B_PARAMS = ModelParams(
    dim=params_1b["dim"],
    n_layers=params_1b["n_layers"],
    n_heads=params_1b["n_heads"],
    n_kv_heads=params_1b["n_kv_heads"],
    head_dim=params_1b["dim"] // params_1b["n_heads"],
    max_seq_len=params_1b["max_seq_len"],
    rope_theta=params_1b["rope_theta"],
    use_scaled_rope=params_1b["use_scaled_rope"],
    distributed=DistributedConfig(
        tensor_parallel_size=1,
        world_size=1,
        layers_per_gpu=params_1b["n_layers"],
        hidden_per_gpu=params_1b["dim"]
    )
)

LLAMA_70B_PARAMS = ModelParams(
    dim=params_70b["dim"],
    n_layers=params_70b["n_layers"],
    n_heads=params_70b["n_heads"],
    n_kv_heads=params_70b["n_kv_heads"],
    head_dim=params_70b["dim"] // params_70b["n_heads"],
    max_seq_len=params_70b["max_seq_len"],
    rope_theta=params_70b["rope_theta"],
    use_scaled_rope=params_70b["use_scaled_rope"],
    distributed=DistributedConfig(
        tensor_parallel_size=8,
        world_size=8,
        layers_per_gpu=10,  # 80 layers / 8 GPUs
        hidden_per_gpu=1024  # 8192 / 8 GPUs
    )
)

def get_model_params(model_size: str = "1B") -> ModelParams:
    """Helper function to get the right model parameters"""
    if model_size == "1B":
        return LLAMA_1B_PARAMS
    elif model_size == "70B":
        return LLAMA_70B_PARAMS
    else:
        raise ValueError(f"Unknown model size: {model_size}")