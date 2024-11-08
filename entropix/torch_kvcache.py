import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class KVCache(nn.Module):
    def __init__(
        self,
        layers: int,
        bsz: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device
    ):
        super(KVCache, self).__init__()
        # Initialize k and v buffers with shapes optimized for GQA (8 KV heads)
        self.register_buffer(
            'k',
            torch.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device
            )
        )
        self.register_buffer(
            'v',
            torch.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device
            )
        )
        self.max_seq_len = max_seq_len

    @classmethod
    def new(
        cls,
        layers: int,
        bsz: int,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device
    ) -> 'KVCache':
        """Creates a new KVCache instance."""
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim, device)

    def update(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        layer_idx: int,
        cur_pos: int,
        n_rep: int
    ):
        """
        Updates cache with new key/value tensors, handling GQA pattern.
        
        Args:
            xk: New key tensor (bsz, seqlen, n_kv_heads, head_dim)
            xv: New value tensor (bsz, seqlen, n_kv_heads, head_dim)  
            layer_idx: Layer index
            cur_pos: Current sequence position
            n_rep: Number of times to repeat KV heads (typically 8 for 64q/8kv)
        """
        if cur_pos >= self.max_seq_len:
            return None, None, self

        # Convert to cache dtype
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)

        # Update cache at current position
        insert_len = xk.size(1)
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len] = xv

        if cur_pos == 0:
            # Initial position: repeat new KV tensors for GQA pattern
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            # Use cached keys/values and repeat for GQA
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        return keys, values, self

    def clear(self):
        """Resets the KV cache."""
        self.k.zero_()
        self.v.zero_()