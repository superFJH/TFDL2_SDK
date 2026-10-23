import math
import threading
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import _tensor, _attention_tensors, _attention_geometry, _fused_scale, _linear_fp32, _module_autocast


class _PackedModule(nn.Module):
    def __init__(self, weight, bias, runtime):
        super().__init__()
        self.runtime = runtime
        self._packing_lock = threading.RLock()
        self._packed, self._version = None, None
        self._weight_shape = tuple(weight.shape)
        # Keep canonical CPU weights for state_dict. Never serialize physical
        # addresses. Copies are independent of the source module and mutable
        # checkpoints; native packing takes a second immutable snapshot.
        with torch.inference_mode(False), torch.no_grad():
            self.register_buffer("weight", _tensor(weight.detach()).clone())
            self.register_buffer("bias", None if bias is None else _tensor(bias.detach()).clone())
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError("bias shape must equal output channels")
        self.eval()

    def prepare(self):
        with self._packing_lock:
            if tuple(self.weight.shape) != self._weight_shape:
                raise ValueError("reloaded weight shape differs from this module's geometry")
            version = (id(self.weight), self.weight.data_ptr(), self.weight._version)
            if self._packed is None or self._version != version:
                packed = self._pack()
                self._packed, self._version = packed, version
            return self._packed

    def close(self):
        with self._packing_lock:
            self._packed, self._version = None, None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        with self._packing_lock:
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
            self.close()  # Repack lazily on first use or explicitly prepare().

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=self.weight.dtype, device="cpu"))
        if probe.dtype not in (torch.float16, torch.float32) or probe.device.type != "cpu":
            raise TypeError("NPU modules support CPU float16/float32 weights; .to(device) is not a device backend")
        self.close()
        return super()._apply(fn, recurse)


class NpuLinear(_PackedModule):
    """Inference-only replacement for a loaded torch.nn.Linear."""
    def __init__(self, weight, bias=None, *, runtime):
        if weight.ndim != 2 or min(weight.shape) <= 0:
            raise ValueError("Linear weight must be [out_features, in_features]")
        super().__init__(weight, bias, runtime)
        self.out_features, self.in_features = weight.shape
        self.prepare()

    @classmethod
    def from_linear(cls, module, *, runtime):
        if type(module) is not nn.Linear:
            raise TypeError("from_linear accepts plain torch.nn.Linear; adapt subclasses explicitly")
        return cls(module.weight, module.bias, runtime=runtime)

    def _pack(self):
        return self.runtime.pack(self.weight)

    def forward(self, x):
        x, = _module_autocast(_tensor(x, contiguous=False))
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("Linear input feature count differs from weight")
        packed = self.prepare()  # Retain through completion even during reload.
        y = self.runtime.linear(x, packed, self.out_features)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def forward_fp32(self, x):
        """Explicit alias for direct FP32 input quantization and FP32 output."""
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("Linear input feature count differs from weight")
        packed = self.prepare()
        y = _linear_fp32(x, self.runtime.id, packed.handle, self.out_features)
        return y if self.bias is None else y + self.bias.to(y.dtype)


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2 or not all(isinstance(x, int) for x in value):
        raise ValueError("expected integer or pair of integers")
    return tuple(value)


class NpuConv2d(_PackedModule):
    """1x1 uses NCHW per-pixel symmetric quantization; spatial uses native Conv or GEMM."""
    def __init__(self, weight, bias=None, *, stride=1, padding=0, dilation=1, groups=1, runtime):
        if weight.ndim != 4 or min(weight.shape) <= 0:
            raise ValueError("Conv2d weight must be nonempty OIHW")
        stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
        if groups != 1 or dilation != (1, 1):
            raise NotImplementedError("native Conv2d currently supports groups=1, dilation=1")
        if stride[0] != stride[1] or not 0 < stride[0] < 8 or min(padding) < 0 or max(padding) >= 8:
            raise NotImplementedError("native Conv2d requires equal strides 1..7 and padding 0..7")
        self.stride, self.padding = stride, padding
        self.out_channels, self.in_channels, kh, kw = weight.shape
        self.kernel_size, self.pointwise = (kh, kw), (kh, kw) == (1, 1)
        self.spatial_implementation = getattr(runtime, "spatial_convolution", "native")
        reduction_limit = (2**31-1)//(255*255) if (kh, kw) == (3, 3) else 4608
        if not self.pointwise and (max(kh, kw) >= 8 or self.in_channels * kh * kw > reduction_limit):
            raise NotImplementedError("native Conv2d requires kernels 1..7; 3x3 Cin*Kh*Kw <=33025 (INT32 accumulator headroom), other kernels <=4608")
        super().__init__(weight, bias, runtime)
        self.prepare()

    @classmethod
    def from_conv2d(cls, module, *, runtime):
        if type(module) is not nn.Conv2d or module.padding_mode != "zeros" or isinstance(module.padding, str):
            raise NotImplementedError("from_conv2d requires plain Conv2d with numeric zero padding")
        return cls(module.weight, module.bias, stride=module.stride, padding=module.padding,
                   dilation=module.dilation, groups=module.groups, runtime=runtime)

    def _pack(self):
        if self.pointwise or self.spatial_implementation == "gemm":
            return self.runtime.pack(self.weight.flatten(1))
        ph, pw = self.padding
        return self.runtime.pack(self.weight, spatial=(self.stride[0], ph, ph, pw, pw))

    def forward_fp32(self, x):
        """Explicit alias; forward() automatically selects the same FP32 path."""
        if x.dtype != torch.float32:raise TypeError("forward_fp32 expects float32")
        with torch.autocast('cpu', enabled=False):return self.forward(x)

    def forward(self, x):
        x, = _module_autocast(_tensor(x, contiguous=False))
        if x.ndim != 4 or min(x.shape) <= 0 or x.shape[1] != self.in_channels:
            raise ValueError("Conv2d expects nonempty NCHW with matching input channels")
        h, w = x.shape[-2:]
        ph, pw = self.padding
        oh = (h + 2 * ph - self.kernel_size[0]) // self.stride[0] + 1
        ow = (w + 2 * pw - self.kernel_size[1]) // self.stride[1] + 1
        if min(oh, ow) <= 0:
            raise ValueError("Conv2d output has nonpositive size")
        packed = self.prepare()
        if self.pointwise:
            y = self.runtime.pointwise(x, packed, self.out_channels, self.stride[0], self.padding)
        elif self.spatial_implementation == "gemm":
            # Materialize only a band of output rows. Padding once preserves
            # the exact official convolution coordinates at every tile boundary.
            padded = F.pad(x, (pw, pw, ph, ph)) if ph or pw else x
            y = x.new_empty((x.shape[0], self.out_channels, oh, ow))
            rows_per_tile = max(1, 1024 // ow)
            kh, kw = self.kernel_size
            for batch in range(x.shape[0]):
                for first in range(0, oh, rows_per_tile):
                    last = min(oh, first + rows_per_tile)
                    begin = first * self.stride[0]
                    end = (last - 1) * self.stride[0] + kh
                    band = padded[batch:batch+1, :, begin:end, :]
                    patches = F.unfold(band, (kh, kw), stride=self.stride).transpose(1, 2).contiguous()
                    result = self.runtime.linear(patches, packed, self.out_channels)
                    y[batch:batch+1, :, first:last, :] = result.transpose(1, 2).reshape(
                        1, self.out_channels, last-first, ow)
        else:
            y = self.runtime.conv(x, packed, self.out_channels, oh, ow)
        return y if self.bias is None else y + self.bias.to(y.dtype)[None, :, None, None]


class NpuConvTranspose2d(_PackedModule):
    """Native stride-2 deconvolution; canonical state_dict weights remain IOHW."""
    def __init__(self, weight, bias=None, *, stride=2, padding=0, output_padding=0,
                 groups=1, dilation=1, runtime):
        if weight.ndim != 4 or min(weight.shape) <= 0:
            raise ValueError("ConvTranspose2d weight must be nonempty IOHW")
        self.stride, self.padding = _pair(stride), _pair(padding)
        self.output_padding, self.dilation = _pair(output_padding), _pair(dilation)
        self._output_packed = {}
        self.in_channels, self.out_channels, kh, kw = weight.shape
        self.kernel_size, self.groups = (kh, kw), groups
        if self.stride != (2, 2):
            raise NotImplementedError("native ConvTranspose2d requires stride=2")
        if groups != 1 or self.dilation != (1, 1):
            raise NotImplementedError("native ConvTranspose2d requires groups=1, dilation=1")
        if max(kh, kw) >= 8 or self.in_channels*kh*kw > 4608:
            raise NotImplementedError("native ConvTranspose2d requires kernels 1..7 and Cin*Kh*Kw <=4608")
        if not all(0 <= p < k for p, k in zip(self.padding, self.kernel_size)) or not all(p in (0, 1) for p in self.output_padding):
            raise NotImplementedError("native ConvTranspose2d requires padding smaller than kernel and output_padding=0..1")
        # _PackedModule's ordinary Conv bias check uses weight.shape[0].
        if bias is not None and bias.shape != (self.out_channels,):
            raise ValueError("bias shape must equal output channels")
        super().__init__(weight, None, runtime)
        if bias is not None:
            with torch.inference_mode(False), torch.no_grad():
                self.bias = _tensor(bias.detach()).clone()
        self.prepare()

    @classmethod
    def from_conv_transpose2d(cls, module, *, runtime):
        if type(module) is not nn.ConvTranspose2d or module.padding_mode != "zeros":
            raise NotImplementedError("from_conv_transpose2d requires plain ConvTranspose2d with zero padding")
        return cls(module.weight, module.bias, stride=module.stride, padding=module.padding,
                   output_padding=module.output_padding, groups=module.groups, dilation=module.dilation, runtime=runtime)

    def close(self):
        with self._packing_lock:
            self._output_packed.clear()
            super().close()

    def _pack(self):
        return self.runtime.pack_transposed(self.weight, self.stride[0], self.padding, self.output_padding)

    def forward(self, x, output_size=None):
        x, = _module_autocast(_tensor(x, contiguous=False))
        if x.ndim != 4 or min(x.shape) <= 0 or x.shape[1] != self.in_channels:
            raise ValueError("ConvTranspose2d expects nonempty NCHW with matching input channels")
        if x.shape[-1] > 512:
            raise NotImplementedError("native ConvTranspose2d input width must be <=512")
        base = tuple((x.shape[d+2]-1)*2-2*self.padding[d]+self.kernel_size[d] for d in range(2))
        if min(base)<=0:raise NotImplementedError("native ConvTranspose2d requires a positive base output size")
        op = self.output_padding
        if output_size is not None:
            shape = tuple(output_size)
            if len(shape) not in (2, 4):raise ValueError("output_size must have 2 or 4 dimensions")
            if len(shape) == 4 and shape[:2] != (x.shape[0], self.out_channels):
                raise ValueError("output_size batch/channels differ")
            op = tuple(shape[-2+d]-base[d] for d in range(2))
            if any(p not in (0, 1) for p in op):raise ValueError("output_size is outside valid stride-2 sizes")
        oh, ow = (base[d]+op[d] for d in range(2))
        if min(oh, ow) <= 0:raise ValueError("ConvTranspose2d output has nonpositive size")
        packed = self.prepare()
        if op != self.output_padding:
            with self._packing_lock:
                entry = self._output_packed.get(op)
                if entry is None or entry[0] != self._version:
                    entry = (self._version, self.runtime.pack_transposed(self.weight, 2, self.padding, op))
                    self._output_packed[op] = entry
                packed = entry[1]
        y = self.runtime.conv_transpose(x, packed, self.out_channels, oh, ow)
        return y if self.bias is None else y+self.bias.to(y.dtype)[None, :, None, None]


class NpuAttention(nn.Module):
    """SDPA core: [B,H,L,D], no projections; both GEMMs use shared runtime.

    auto: native INT32 fusion for unmasked/causal NPU calls, otherwise bounded
    Torch softmax compatibility path (warns on NPU fallback). fused: never
    falls back. torch: explicitly use the old tiled runtime GEMM/Torch path.
    auto also honors runtime.attention_policy='hybrid'/'cpu', using original
    Torch CPU SDPA for the selected geometries. fused always requires NPU.
    query_tile controls only the compatibility path; runtime owns fused tiling.
    Causal masking uses upper-left alignment, as PyTorch SDPA does.
    """
    def __init__(self, *, runtime, query_tile=128, implementation="auto"):
        super().__init__()
        if not isinstance(query_tile, int) or query_tile <= 0 or query_tile > 1024:
            raise ValueError("query_tile must be 1..1024")
        if implementation not in ("auto", "fused", "torch"):
            raise ValueError("implementation must be auto, fused or torch")
        self.runtime, self.query_tile = runtime, query_tile
        self.implementation = implementation
        self.eval()

    def forward(self, query, key, value, attn_mask=None, *, is_causal=False, scale=None, dropout_p=0.0, enable_gqa=False):
        if dropout_p != 0:
            raise NotImplementedError("NpuAttention is inference-only; dropout_p must be zero")
        q, k, v = _module_autocast(*_attention_tensors(query, key, value, contiguous=False))
        b, heads, kv_heads, length, keys, dim, out_dim = _attention_geometry(q, k, v, enable_gqa)
        scale = dim ** -0.5 if scale is None else float(scale)
        if not math.isfinite(scale):
            raise ValueError("non-finite attention scale")
        eligible = attn_mask is None and _fused_scale(scale) and max(keys, dim, out_dim) <= 16384
        if self.implementation == "fused":
            if attn_mask is not None:
                raise NotImplementedError("fused attention supports no mask or is_causal; arbitrary masks require implementation='torch'")
            return self.runtime.attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if self.implementation == "auto" and self.runtime.backend == "npu":
            if self.runtime.attention_backend(q,k,v,attn_mask=attn_mask,is_causal=is_causal) == "torch":
                return F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,is_causal=is_causal,
                                                     scale=scale,enable_gqa=enable_gqa)
            if eligible:
                return self.runtime.attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            warnings.warn("NpuAttention is using tiled Torch softmax for this mask/shape/scale; "
                          "use implementation='fused' to disallow fallback", RuntimeWarning, stacklevel=2)
        mask = None
        if attn_mask is not None:
            if attn_mask.device.type != "cpu" or attn_mask.requires_grad or (attn_mask.dtype != torch.bool and not attn_mask.is_floating_point()):
                raise TypeError("Attention mask must be a CPU bool/float inference tensor")
            if is_causal:
                raise ValueError("provide attn_mask or is_causal, not both")
            mask = torch.broadcast_to(attn_mask, (b, heads, length, keys))
            if mask.is_floating_point() and (torch.isnan(mask).any() or torch.isposinf(mask).any()):
                raise ValueError("Attention additive mask cannot contain NaN/+inf")
        result = torch.empty((b, heads, length, out_dim), dtype=q.dtype)
        for batch in range(b):
            for head in range(heads):
                # Per-invocation snapshots; no stale K/V cache keyed by pointer.
                kv_head = head // (heads // kv_heads)
                packed_k = self.runtime.pack(k[batch, kv_head], dynamic=True)
                packed_v = self.runtime.pack(v[batch, kv_head].T.contiguous(), dynamic=True)
                for first in range(0, length, self.query_tile):
                    last = min(first + self.query_tile, length)
                    scores = self.runtime.linear(q[batch, head, first:last], packed_k, keys).float() * scale
                    if is_causal:
                        visible = torch.arange(keys)[None, :] <= torch.arange(first, last)[:, None]
                        scores.masked_fill_(~visible, -torch.inf)
                    if mask is not None:
                        m = mask[batch, head, first:last]
                        if m.dtype == torch.bool:
                            scores.masked_fill_(~m, -torch.inf)
                        else:
                            scores += m.float()
                    # PyTorch SDPA returns zeros for a fully masked query row.
                    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
                    scores.masked_fill_(empty, 0)
                    probability = torch.softmax(scores, dim=-1).masked_fill_(empty, 0).to(q.dtype)
                    result[batch, head, first:last] = self.runtime.linear(probability, packed_v, out_dim)
        return result


def replace_linear(model, *, runtime, predicate=None):
    """Replace plain nn.Linear children after checkpoint loading. Explicit opt-in.

    Does not intercept F.linear, SDPA, custom subclasses, hooks or tied weights.
    """
    replacements = []
    def visit(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if type(child) is nn.Linear and (predicate is None or predicate(path, child)):
                if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks:
                    raise ValueError(f"{path} has hooks; adapt it explicitly")
                replacements.append((parent, name, NpuLinear.from_linear(child, runtime=runtime)))
            else:
                visit(child, path)
    visit(model, "")  # Prepare everything before mutating the model tree.
    for parent, name, module in replacements:
        setattr(parent, name, module)
    return model
