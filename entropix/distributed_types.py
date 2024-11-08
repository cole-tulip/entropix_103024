from dataclasses import dataclass
import torch.distributed as dist

@dataclass
class ProcessGroups:
    """Container for process groups used in distributed training."""
    attn: dist.ProcessGroup
    ffn: dist.ProcessGroup