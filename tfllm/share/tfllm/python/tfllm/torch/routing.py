"""Bounded mask normalization and observable, inference-only SDPA routing."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .optimize import Unsupported
from .runtime import _attention_geometry, _fused_scale


@dataclass
class IntervalMask:
    bounds: torch.Tensor  # [B,L,2], never [B,H,L,S]


@dataclass
class DeferredMask:
    reason: str
    materialize: object


def interval_tensor(mask, b, heads, length, keys, causal=False):
    if isinstance(mask, IntervalMask):
        bounds = mask.bounds
        if bounds.shape != (b, length, 2):
            raise Unsupported("compact mask geometry differs from Q/K")
        if causal:
            bounds = bounds.clone()
            bounds[..., 1] = torch.minimum(bounds[..., 1], torch.arange(1, length + 1))
            bounds[..., 0] = torch.minimum(bounds[..., 0], bounds[..., 1])
        return bounds
    if mask is None:
        bounds = torch.empty((b, length, 2), dtype=torch.int64)
        bounds[..., 0] = 0
        bounds[..., 1] = torch.arange(1, length + 1).clamp(max=keys) if causal else keys
        return bounds
    if mask.device.type != "cpu" or mask.requires_grad:
        raise Unsupported("mask must be a CPU inference tensor")
    try:
        view = torch.broadcast_to(mask, (b, heads, length, keys))
    except RuntimeError as error:
        raise Unsupported("mask is not broadcastable to [B,H,L,S]") from error
    # Per-head different masks need a richer native ABI. Compare without
    # constructing an expanded mask or extra [H,L,S] boolean activation.
    for h in range(1, heads):
        if not torch.equal(view[:, h], view[:, 0]):
            raise Unsupported("head-dependent attention masks are unsupported")
    bounds = torch.empty((b, length, 2), dtype=torch.int64)
    for batch in range(b):
        for first in range(0, length, 32):
            row = view[batch, 0, first:first + 32]
            if row.dtype == torch.bool:
                visible = row
            elif row.is_floating_point() and bool(((row == 0) | torch.isneginf(row)).all()):
                visible = row == 0
            else:
                raise Unsupported("fusion requires bool or exactly 0/-inf mask; additive biases remain original")
            count = visible.sum(-1)
            begin = visible.to(torch.uint8).argmax(-1)
            end = keys - visible.flip(-1).to(torch.uint8).argmax(-1)
            begin = torch.where(count == 0, 0, begin)
            end = torch.where(count == 0, 0, end)
            if bool((end - begin != count).any()):
                raise Unsupported("non-contiguous visible keys are unsupported")
            if causal:
                end = torch.minimum(end, torch.arange(first + 1, first + 1 + len(end)))
                begin = torch.minimum(begin, end)
            bounds[batch, first:first + len(end), 0] = begin
            bounds[batch, first:first + len(end), 1] = end
    return bounds


def materialize_mask(mask, keys):
    if isinstance(mask, DeferredMask):
        return mask.materialize()
    if isinstance(mask, IntervalMask):
        columns = torch.arange(keys)
        return ((columns >= mask.bounds[..., 0, None]) & (columns < mask.bounds[..., 1, None]))[:, None]
    return mask


class AttentionRoute:
    def __init__(self, state, path):
        self.state, self.path = state, path

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        def original():
            mask = materialize_mask(attn_mask, key.shape[-2])
            return F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        try:
            if torch.is_grad_enabled() or dropout_p != 0:
                raise Unsupported("attention requires no_grad()/inference_mode() and dropout=0")
            if any(x.device.type != "cpu" or x.dtype != torch.float16 for x in (query, key, value)):
                raise Unsupported("attention activations must be CPU float16")
            try:
                b, heads, _, length, keys, dim, out_dim = _attention_geometry(query, key, value, enable_gqa)
            except ValueError as error:
                raise Unsupported(str(error)) from error
            scale = dim ** -.5 if scale is None else float(scale)
            if max(keys, dim, out_dim) > 16384 or not _fused_scale(scale):
                raise Unsupported("fused attention shape/scale is unsupported")
            if isinstance(attn_mask, DeferredMask):
                raise Unsupported(attn_mask.reason)
            bounds = None if attn_mask is None else interval_tensor(attn_mask, b, heads, length, keys, is_causal)
        except Unsupported as error:
            return self.state.fallback(self.path, str(error), original)
        if self.state.runtime.backend == "cpu":
            self.state.report.record(self.path, "cpu_reference")
            return original()
        if bounds is None:
            result = self.state.runtime.attention(query, key, value, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        else:
            result = self.state.runtime.attention_intervals(query, key, value, bounds, scale=scale, enable_gqa=enable_gqa)
        self.state.report.record(self.path, "npu_fused")
        return result
