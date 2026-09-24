"""Explicit block-mask adapter. Does not invent FlashVSR's top-k/window policy."""
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class BlockSparseMask:
    """CSR rows flatten [B,Hq,ceil(L/Bq)], columns name KV blocks.

    Rows must be sorted/unique; empty rows return zero. Tensor contents are
    mutable between calls, never during a call. Native code validates each call.
    """
    row_offsets: torch.Tensor
    column_indices: torch.Tensor
    query_block_size: int = 128
    kv_block_size: int = 128
    _dense_shape: tuple = field(default=None, repr=False)

    @classmethod
    def from_dense(cls, block_mask, *, query_block_size=128, kv_block_size=128):
        """Convert CPU bool [B,Hq,Qblocks,KVblocks], NOT a token-level mask."""
        if block_mask.device.type != 'cpu' or block_mask.dtype != torch.bool or block_mask.ndim != 4 or min(block_mask.shape) <= 0:
            raise ValueError("block_mask must be nonempty CPU bool [B,Hq,Qblocks,KVblocks]")
        rows = block_mask.reshape(-1, block_mask.shape[-1])
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64), rows.sum(-1, dtype=torch.int64).cumsum(0)))
        columns = rows.nonzero(as_tuple=True)[1].contiguous()
        return cls(offsets, columns, query_block_size, kv_block_size, tuple(block_mask.shape))


class NpuBlockSparseAttention(nn.Module):
    """Inference-only CSR core; native gather + fused QK/softmax/AV, no fallback."""
    def __init__(self, *, runtime):
        super().__init__()
        self.runtime = runtime
        self.eval()

    def forward(self, query, key, value, mask, *, scale=None, enable_gqa=False):
        if not isinstance(mask, BlockSparseMask):
            raise TypeError("mask must be BlockSparseMask; use from_dense for a block-level bool mask")
        if mask._dense_shape is not None:
            bq, bk = mask.query_block_size, mask.kv_block_size
            if type(bq) is not int or type(bk) is not int or bq <= 0 or bk <= 0:
                raise ValueError("block sizes must be positive integers")
            if query.ndim != 4 or key.ndim != 4:
                raise ValueError("block attention Q/K must be [B,H,L,D]")
            expected = (query.shape[0], query.shape[1], (query.shape[2]+bq-1)//bq, (key.shape[2]+bk-1)//bk)
            if mask._dense_shape != expected:
                raise ValueError(f"block_mask shape {mask._dense_shape} differs from Q/K blocks {expected}")
        return self.runtime.attention_blocks(query, key, value, mask.row_offsets, mask.column_indices,
            query_block_size=mask.query_block_size, kv_block_size=mask.kv_block_size,
            scale=scale, enable_gqa=enable_gqa)
