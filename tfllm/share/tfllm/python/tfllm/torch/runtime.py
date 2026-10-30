"""Synchronous native calls over borrowed CPU tensor storage; no NumPy bridge."""
import ctypes as C
import itertools
import math
import os
from pathlib import Path
import threading
import weakref

import torch

_contexts = weakref.WeakValueDictionary()
_ids = itertools.count(1)


def _context(identity):
    runtime = _contexts.get(identity)
    if runtime is None:
        raise RuntimeError("TFLLM runtime is closed or unavailable in this process")
    return runtime


def _tensor(x, *, contiguous=True):
    if x.device.type != "cpu" or x.dtype not in (torch.float16, torch.float32):
        raise TypeError("TFLLM expects CPU float16 or float32 tensors")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("TFLLM is inference-only; use torch.inference_mode()")
    return x.contiguous() if contiguous else x


def _external_dtype(context):
    if not hasattr(context._lib, 'tfllm_torch_external_dtypes') or context._lib.tfllm_torch_external_dtypes() != 1:
        raise RuntimeError("rebuild/install libtfllm-torch and Python together for direct FP32 dtype support")


def _attention_tensors(query, key, value, *, contiguous=True):
    tensors = tuple(_tensor(x, contiguous=contiguous) for x in (query, key, value))
    if any(x.dtype != query.dtype for x in tensors):
        raise TypeError("attention Q/K/V must have the same dtype")
    return tensors


def _module_autocast(*tensors):
    """Module forwards honor the caller's requested output dtype."""
    if not torch.is_autocast_enabled('cpu'):
        return tensors
    dtype = torch.get_autocast_dtype('cpu')
    if dtype not in (torch.float16, torch.float32):
        raise TypeError('TFLLM CPU autocast currently supports float16/float32')
    return tuple(x if x.dtype == dtype else x.to(dtype=dtype, memory_format=torch.contiguous_format) for x in tensors)


def _attention_geometry(q, k, v, enable_gqa):
    if any(t.ndim != 4 or min(t.shape) <= 0 for t in (q, k, v)):
        raise ValueError("Attention requires nonempty [batch, heads, sequence, channels]")
    b, heads, length, dim = q.shape
    if (k.shape[0] != b or v.shape[:2] != k.shape[:2] or k.shape[-1] != dim
            or k.shape[2] != v.shape[2] or heads % k.shape[1]
            or (not enable_gqa and heads != k.shape[1])):
        raise ValueError("Attention batch/head/dimension mismatch; GQA requires enable_gqa=True and Hq % Hkv == 0")
    return b, heads, k.shape[1], length, k.shape[2], dim, v.shape[-1]


def _fused_scale(scale):
    return math.isfinite(scale) and 0 < C.c_float(scale).value < math.inf


def _library(path):
    if path is None:
        path = os.environ.get("TFLLM_TORCH_LIBRARY")
    if path is None:
        module = Path(__file__).resolve()
        # pip: site-packages/tfllm/torch plus sibling tfllm_runtime/sdk.
        roots = [module.parents[2] / "tfllm_runtime" / "sdk"]
        # CMake: tfllm/share/tfllm/python/tfllm/torch/runtime.py. Only
        # accept this exact layout, not an arbitrary ancestor's lib directory.
        if len(module.parents) > 5:
            root = module.parents[5]
            if root / "share/tfllm/python/tfllm/torch/runtime.py" == module:
                roots.insert(0, root)
        for root in roots:
            for name in ("libtfllm-torch.so", "libtfllm-torch.dylib"):
                candidate = root / "lib" / name
                if candidate.is_file():
                    path = candidate
                    break
            if path is not None:
                break
    if path is None:
        raise RuntimeError("Set TFLLM_TORCH_LIBRARY to libtfllm-torch; build with TFLLM_WITH_TORCH=ON")
    lib = C.CDLL(str(path))  # CDLL releases the GIL while native workers execute.
    lib.tfllm_torch_abi.restype, lib.tfllm_torch_abi.argtypes = C.c_int, []
    if lib.tfllm_torch_abi() != 2:
        raise RuntimeError("TFLLM Torch native ABI mismatch; rebuild/install libtfllm-torch and Python together (ABI 2)")
    signatures = {
        "abi": (C.c_int, []), "has_npu": (C.c_int, []), "error": (C.c_char_p, []),
        "create": (C.c_int, [C.c_int, C.c_int, C.c_double, C.POINTER(C.c_size_t), C.POINTER(C.c_void_p)]),
        "destroy": (C.c_int, [C.c_void_p]),
        "weight": (C.c_int, [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t, C.c_int, C.POINTER(C.c_uint64)]),
        "release": (C.c_int, [C.c_void_p, C.c_uint64]),
        "linear": (C.c_int, [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p] + [C.c_size_t] * 3),
        "conv_weight": (C.c_int, [C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_uint64)]),
        "conv": (C.c_int, [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p] + [C.c_size_t] * 7),
        "attention": (C.c_int, [C.c_void_p] * 5 + [C.POINTER(C.c_size_t), C.c_int, C.c_float]),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, "tfllm_torch_" + name)
        fn.restype, fn.argtypes = result, args
    if hasattr(lib, "tfllm_torch_memory_stats"):
        lib.tfllm_torch_memory_stats.restype = C.c_int
        lib.tfllm_torch_memory_stats.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
    for name, base in (("weight_f32", "weight"), ("linear_f32", "linear"),
                       ("conv_weight_f32", "conv_weight"), ("conv_f32", "conv"), ("attention_f32", "attention")):
        if hasattr(lib, "tfllm_torch_" + name):
            fn = getattr(lib, "tfllm_torch_" + name)
            fn.restype, fn.argtypes = signatures[base]
    for name in ("conv_transpose_weight", "conv_transpose_weight_f32"):
        if hasattr(lib, "tfllm_torch_" + name):
            fn = getattr(lib, "tfllm_torch_" + name)
            fn.restype, fn.argtypes = signatures["conv_weight"]
    if hasattr(lib, "tfllm_torch_external_dtypes"):
        lib.tfllm_torch_external_dtypes.restype, lib.tfllm_torch_external_dtypes.argtypes = C.c_int, []
    if hasattr(lib, "tfllm_torch_linear_fp32"):
        lib.tfllm_torch_linear_fp32.restype, lib.tfllm_torch_linear_fp32.argtypes = signatures["linear"]
    if hasattr(lib, "tfllm_torch_pointwise"):
        lib.tfllm_torch_pointwise.restype = C.c_int
        lib.tfllm_torch_pointwise.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t)]
    if hasattr(lib, "tfllm_torch_pointwise_fp32"):
        lib.tfllm_torch_pointwise_fp32.restype = C.c_int
        lib.tfllm_torch_pointwise_fp32.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t)]
    if hasattr(lib, "tfllm_torch_window_fp32"):
        lib.tfllm_torch_window_fp32.restype = C.c_int
        lib.tfllm_torch_window_fp32.argtypes = [C.c_void_p] * 5 + [C.POINTER(C.c_size_t)] * 2 + [C.c_float]
    if hasattr(lib, "tfllm_torch_window_f32"):
        lib.tfllm_torch_window_f32.restype = C.c_int
        lib.tfllm_torch_window_f32.argtypes = [C.c_void_p] * 5 + [C.POINTER(C.c_size_t)] * 2 + [C.c_float]
    if hasattr(lib, "tfllm_torch_window_f16"):
        lib.tfllm_torch_window_f16.restype = C.c_int
        lib.tfllm_torch_window_f16.argtypes = [C.c_void_p] * 5 + [C.POINTER(C.c_size_t)] * 2 + [C.c_float]
    if hasattr(lib, "tfllm_torch_attention_intervals"):
        lib.tfllm_torch_attention_intervals.restype = C.c_int
        lib.tfllm_torch_attention_intervals.argtypes = [C.c_void_p] * 5 + [C.POINTER(C.c_size_t), C.c_void_p, C.c_float]
    if hasattr(lib, "tfllm_torch_attention_blocks"):
        lib.tfllm_torch_attention_blocks.restype = C.c_int
        lib.tfllm_torch_attention_blocks.argtypes = ([C.c_void_p] * 5 + [C.POINTER(C.c_size_t),
            C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_float])
    for base in ("attention_intervals", "attention_blocks"):
        if hasattr(lib, "tfllm_torch_" + base + "_f32"):
            fn, original = getattr(lib, "tfllm_torch_" + base + "_f32"), getattr(lib, "tfllm_torch_" + base)
            fn.restype, fn.argtypes = original.restype, original.argtypes
    for name in ("attention_strided_f16", "attention_strided_f32"):
        if hasattr(lib,"tfllm_torch_"+name):
            fn=getattr(lib,"tfllm_torch_"+name)
            fn.restype,fn.argtypes=C.c_int,[C.c_void_p]*5+[C.POINTER(C.c_size_t)]*2+[C.c_int,C.c_float]
    for name in ("activation_f16", "activation_f32"):
        if hasattr(lib, "tfllm_torch_" + name):
            fn = getattr(lib, "tfllm_torch_" + name)
            fn.restype, fn.argtypes = C.c_int, [C.c_void_p]*3 + [C.c_size_t, C.c_int]
    if hasattr(lib, "tfllm_torch_activation_version"):
        lib.tfllm_torch_activation_version.restype, lib.tfllm_torch_activation_version.argtypes = C.c_int, []
    if hasattr(lib, "tfllm_torch_create_fast"):
        lib.tfllm_torch_create_fast.restype = C.c_int
        lib.tfllm_torch_create_fast.argtypes = [C.c_int,C.c_int,C.c_double,C.POINTER(C.c_size_t),C.c_int,C.POINTER(C.c_void_p)]
    if hasattr(lib, "tfllm_torch_attention_fast_version"):
        lib.tfllm_torch_attention_fast_version.restype, lib.tfllm_torch_attention_fast_version.argtypes = C.c_int, []
    for name in ("attention_mask_f16","attention_mask_f32"):
        if hasattr(lib,"tfllm_torch_"+name):
            fn=getattr(lib,"tfllm_torch_"+name)
            fn.restype,fn.argtypes=C.c_int,[C.c_void_p]*5+[C.POINTER(C.c_size_t),C.c_void_p,C.POINTER(C.c_size_t),C.c_int,C.c_int,C.c_float]
    return lib


class PackedWeight:
    def __init__(self, runtime, handle):
        self.runtime, self.handle = runtime, handle

    def __del__(self):
        try:
            self.runtime._release(self.handle)
        except Exception:
            pass  # Finalizers must not mask an earlier device failure.


class NpuRuntime:
    """Share a runtime within each process; independent processes may coexist.

    memory_fraction caps the on-demand NPU memory pool (Torch default). It does
    not reserve that fraction at startup. CPU mode is a development oracle.
    """
    def __init__(self, chip=0, *, backend="npu", memory_fraction=0.8, library=None,
                 attention_fuse_workers=16, attention_score_mib=4,
                 attention_pipeline=True, attention_lease_tiles=4,
                 spatial_convolution="native", attention_policy=None, attention_fast=None):
        if backend not in ("npu", "cpu"):
            raise ValueError("backend must be npu or cpu")
        if spatial_convolution not in ("native", "gemm"):
            raise ValueError("spatial_convolution must be native or gemm")
        if attention_policy is None:
            attention_policy=os.environ.get("TFLLM_TORCH_ATTENTION_POLICY","npu")
        if attention_policy not in ("npu", "hybrid", "cpu"):
            raise ValueError("attention_policy must be npu, hybrid or cpu")
        for name, value, limit in (("attention_fuse_workers", attention_fuse_workers, 32),
                                   ("attention_score_mib", attention_score_mib, 16),
                                   ("attention_lease_tiles", attention_lease_tiles, 16)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be 1..{limit}")
        if type(attention_pipeline) is not bool:
            raise TypeError("attention_pipeline must be bool")
        if attention_fast is not None and type(attention_fast) is not bool:
            raise TypeError("attention_fast must be bool or None")
        fast_env=os.environ.get("TFLLM_NPU_ATTENTION_FAST","0")
        if attention_fast is None and fast_env not in ("0","1"):
            raise ValueError("TFLLM_NPU_ATTENTION_FAST must be 0 or 1")
        resolved_fast=(fast_env=="1") if attention_fast is None else attention_fast
        if backend=="cpu":
            if attention_fast is True:raise ValueError("FAST attention requires backend='npu'")
            resolved_fast=False
        self._cv = threading.Condition()
        self._pid = os.getpid()
        self._active, self._closing = 0, False
        self._pointer = C.c_void_p()
        self._lib = _library(library)
        self.backend, self.chip = backend, chip
        self.spatial_convolution = spatial_convolution
        self.attention_policy = attention_policy
        self._attention_fast = resolved_fast
        self.id = next(_ids)
        attention = (C.c_size_t * 4)(attention_fuse_workers, attention_score_mib,
                                    int(attention_pipeline), attention_lease_tiles)
        if attention_fast is not None and not hasattr(self._lib,"tfllm_torch_create_fast"):
            raise RuntimeError("rebuild/install libtfllm-torch for explicit FAST attention selection")
        if self.attention_fast and (not hasattr(self._lib,"tfllm_torch_attention_fast_version")
                or self._lib.tfllm_torch_attention_fast_version()<1):
            raise RuntimeError("rebuild/install libtfllm-torch for general FAST attention")
        if hasattr(self._lib,"tfllm_torch_create_fast"):
            self._check(self._lib.tfllm_torch_create_fast(int(backend=="npu"),chip,memory_fraction,
                attention,int(self.attention_fast),C.byref(self._pointer)))
        else:
            self._check(self._lib.tfllm_torch_create(int(backend=="npu"),chip,memory_fraction,
                attention,C.byref(self._pointer)))
        _contexts[self.id] = self

    @property
    def attention_fast(self):
        """Read-only mode selected at runtime creation; explicit bool overrides env."""
        return self._attention_fast

    def attention_backend(self, query, key, value, *, attn_mask=None, is_causal=False, dtype=None):
        """High-level routing policy; attention() itself always requires NPU.

        Hybrid gates are conservative A77/NPU40T board measurements, not an
        online benchmark or a guarantee on other CPUs/thread configurations.
        No tensor contents/pointers or mutable K/V are cached by this policy.
        """
        if self.backend == "npu" and self.attention_fast:return "npu"
        if self.backend == "cpu" or self.attention_policy == "cpu":
            return "torch"
        if self.attention_policy == "npu":
            return "npu"
        eligible = (query.ndim == key.ndim == value.ndim == 4 and attn_mask is None and not is_causal
                    and query.shape[1] == key.shape[1] == value.shape[1]
                    and 32 <= query.shape[-1] <= 256 and query.shape[-1] == value.shape[-1]
                    and key.shape[-2] <= 16384)
        if not eligible:return "torch"
        execution_dtype=query.dtype if dtype is None else dtype
        large = ((execution_dtype==torch.float16 or query.shape[-1]>=48)
                 and query.shape[-2] >= 1024 and key.shape[-2] >= 1024
                 and query.shape[-2] * key.shape[-2] >= 4 * 1024 * 1024)
        half_window = (execution_dtype==torch.float16
                       and 16 <= query.shape[-2] <= 1024 and 32 <= key.shape[-2] <= 1024
                       and query.shape[0]*query.shape[1]>=32
                       and query.shape[0]*query.shape[1]*query.shape[-2]*key.shape[-2]*query.shape[-1]>=16*1024*1024)
        return "npu" if large or half_window else "torch"

    def _check(self, status):
        if status:
            raise RuntimeError(self._lib.tfllm_torch_error().decode("utf-8", errors="replace"))

    def memory_stats(self):
        """Bytes registered on the chip, process budget, committed pages, and live allocations.

        Committed includes idle cached operands. Live allocations also include
        the separate direct-command cache; registration ownership is per page.
        """
        if not hasattr(self._lib, "tfllm_torch_memory_stats"):
            raise RuntimeError("rebuild the native library to query memory statistics")
        values = (C.c_uint64 * 4)()
        self._call("memory_stats", values)
        return dict(zip(("registered_bytes", "budget_bytes", "committed_bytes", "used_bytes"), values))

    def _call(self, name, *args, allow_unsupported=False):
        if os.getpid() != self._pid:
            raise RuntimeError("create a new NPU runtime after fork; handles cannot cross processes")
        with self._cv:
            if self._closing or not self._pointer.value:
                raise RuntimeError("TFLLM runtime is closed")
            self._active += 1
        try:
            status = getattr(self._lib, "tfllm_torch_" + name)(self._pointer, *args)
            if allow_unsupported and status == 1:
                return False
            self._check(status)
            return True
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def _release(self, handle):
        with self._cv:
            if self._closing or not self._pointer.value:
                return
        self._call("release", handle)

    def pack(self, weight, *, dynamic=False, spatial=None):
        weight = _tensor(weight)
        suffix = "_f32" if weight.dtype == torch.float32 else ""
        if suffix: _external_dtype(self)
        handle = C.c_uint64()
        if spatial is None:
            if weight.ndim != 2 or min(weight.shape) <= 0:
                raise ValueError("matrix weight must be nonempty [out_features, in_features]")
            self._call("weight" + suffix, weight.data_ptr(), *weight.shape, int(dynamic), C.byref(handle))
        else:
            if weight.ndim != 4 or min(weight.shape) <= 0 or len(spatial) != 5 or min(spatial) < 0:
                raise ValueError("spatial weights require OIHW and stride/top/bottom/left/right")
            params = (C.c_size_t * 9)(*weight.shape, *spatial)
            self._call("conv_weight" + suffix, weight.data_ptr(), params, C.byref(handle))
        return PackedWeight(self, handle.value)

    def pack_transposed(self, weight, stride, padding, output_padding):
        weight = _tensor(weight)
        if weight.ndim != 4 or min(weight.shape) <= 0:raise ValueError("ConvTranspose2d weight must be IOHW")
        name = "conv_transpose_weight" + ("_f32" if weight.dtype == torch.float32 else "")
        if not hasattr(self._lib, "tfllm_torch_" + name):
            raise RuntimeError("rebuild libtfllm-torch for ConvTranspose2d support")
        if weight.dtype == torch.float32:_external_dtype(self)
        canonical = weight.permute(1, 0, 2, 3).flip(-1, -2).contiguous()
        params = (C.c_size_t*9)(*canonical.shape, stride, *padding, *output_padding)
        handle = C.c_uint64()
        self._call(name, canonical.data_ptr(), params, C.byref(handle))
        return PackedWeight(self, handle.value)

    def conv_transpose(self, x, weight, out_channels, height, width):
        if weight.runtime is not self:raise ValueError("weight belongs to a different runtime")
        return _conv_transpose(_tensor(x), self.id, weight.handle, out_channels, height, width)

    def linear(self, x, weight, out_features):
        if weight.runtime is not self:
            raise ValueError("weight belongs to a different runtime")
        return _linear(_tensor(x), self.id, weight.handle, out_features)

    def _activation(self, input, kind, *, inplace=False):
        x = _tensor(input)
        name = "activation_f32" if x.dtype == torch.float32 else "activation_f16"
        if not hasattr(self._lib, "tfllm_torch_" + name):
            raise RuntimeError("rebuild/install libtfllm-torch for fast GELU/SiLU support")
        output = torch.empty_like(x)
        self._call(name, x.data_ptr(), output.data_ptr(), x.numel(), kind)
        if inplace:
            # copy_ honors views, overlap checks and Torch's version counter.
            # A raw native write into input storage would bypass those checks.
            input.copy_(output)
            return input
        return output

    def gelu(self, input, *, approximate="none"):
        """CPU TFDL-vector GELU, FP16/FP32; default erf, optional tanh."""
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")
        return self._activation(input, int(approximate == "tanh"))

    def silu(self, input, *, inplace=False):
        """CPU TFDL-vector SiLU; dtype is preserved, including under autocast."""
        return self._activation(input, 2, inplace=inplace)

    def activation(self, input, name, **parameters):
        """Explicit native CPU activation; nondefault Softplus/ELU use Torch in auto routing."""
        from .activation_ops import native_id
        kind = native_id(name, parameters)
        if kind is None:
            raise NotImplementedError("native activation requires Softplus beta=1/threshold=20 or ELU alpha=1")
        if kind >= 3 and (not hasattr(self._lib, "tfllm_torch_activation_version")
                          or self._lib.tfllm_torch_activation_version() < 2):
            raise RuntimeError("rebuild/install libtfllm-torch for extended activation support")
        return self._activation(input, kind, inplace=parameters.get("inplace", False))

    def sigmoid(self, input):return self.activation(input, "sigmoid")
    def tanh(self, input):return self.activation(input, "tanh")
    def mish(self, input, *, inplace=False):return self.activation(input, "mish", inplace=inplace)
    def softplus(self, input, *, beta=1, threshold=20):return self.activation(input, "softplus", beta=beta, threshold=threshold)
    def elu(self, input, *, alpha=1, inplace=False):return self.activation(input, "elu", alpha=alpha, inplace=inplace)
    def selu(self, input, *, inplace=False):return self.activation(input, "selu", inplace=inplace)

    def conv(self, x, weight, out_channels, height, width):
        if weight.runtime is not self:
            raise ValueError("weight belongs to a different runtime")
        return _conv(_tensor(x), self.id, weight.handle, out_channels, height, width)

    def pointwise(self, x, weight, out_channels, stride=1, padding=(0, 0)):
        if weight.runtime is not self:
            raise ValueError("weight belongs to a different runtime")
        x = _tensor(x)
        op = _pointwise_fp32 if x.dtype == torch.float32 else _pointwise
        return op(x, self.id, weight.handle, out_channels, stride, *padding)

    def attention(self, query, key, value, *, is_causal=False, scale=None, enable_gqa=False, attn_mask=None):
        """Strict fused SDPA: no Torch score/probability tensors or fallback."""
        if self.backend != "npu":
            raise NotImplementedError("fused attention requires backend='npu'; CPU oracle uses implementation='torch'")
        q, k, v = _attention_tensors(query, key, value, contiguous=False)
        shape = _attention_geometry(q, k, v, enable_gqa)
        scale = shape[5] ** -0.5 if scale is None else float(scale)
        if attn_mask is not None:
            if not self.attention_fast:raise ValueError("arbitrary attention masks require attention_fast=True")
            return _attention_mask(q,k,v,attn_mask,self.id,is_causal,scale,enable_gqa)
        op = _attention_fp32 if q.dtype == torch.float32 else _attention
        return op(q, k, v, self.id, is_causal, scale, enable_gqa)

    def attention_fp32(self, query, key, value, *, is_causal=False, scale=None, enable_gqa=False):
        """Direct FP32 SDPA; same dispatch as attention() for FP32 Q/K/V."""
        if self.backend != "npu":
            raise NotImplementedError("FP32 attention bridge requires backend='npu'")
        shape = _attention_geometry(query, key, value, enable_gqa)
        scale = shape[5] ** -0.5 if scale is None else float(scale)
        return _attention_fp32(query, key, value, self.id, is_causal, scale, enable_gqa)

    def attention_intervals(self, query, key, value, intervals, *, scale=None, enable_gqa=False):
        """Compact [B,L,2] int64 visible key intervals; empty rows return zero."""
        if self.backend != "npu":
            raise NotImplementedError("fused attention requires backend='npu'")
        if not hasattr(self._lib, "tfllm_torch_attention_intervals"):
            raise RuntimeError("rebuild libtfllm-torch for interval attention support")
        q, k, v = _attention_tensors(query, key, value)
        shape = _attention_geometry(q, k, v, enable_gqa)
        scale = shape[5] ** -0.5 if scale is None else float(scale)
        return _attention_intervals(q, k, v, intervals, self.id, scale, enable_gqa)

    def attention_blocks(self, query, key, value, row_offsets, column_indices, *,
                         query_block_size=128, kv_block_size=128, scale=None, enable_gqa=False):
        """Strict block CSR attention; sorted unique KV block IDs per B/Hq/Qblock.

        CPU int64 offsets/indices. Native code snapshots and validates CSR,
        gathers only selected K/V, and runs one fused attention per query block.
        All masks/KV are invocation-local; no cross-frame mutable tensor cache.
        """
        if self.backend != "npu":
            raise NotImplementedError("fused block attention requires backend='npu'; no implicit CPU fallback")
        if not hasattr(self._lib, "tfllm_torch_attention_blocks"):
            raise RuntimeError("rebuild libtfllm-torch for block sparse attention support")
        for name, block, limit in (("query_block_size", query_block_size, 1024), ("kv_block_size", kv_block_size, 16384)):
            if type(block) is not int or not 1 <= block <= limit:
                raise ValueError(f"{name} must be an integer in 1..{limit}")
        q, k, v = _attention_tensors(query, key, value)
        shape = _attention_geometry(q, k, v, enable_gqa)
        scale = shape[5] ** -0.5 if scale is None else float(scale)
        return _attention_blocks(q, k, v, row_offsets, column_indices, self.id,
                                 query_block_size, kv_block_size, scale, enable_gqa)

    def close(self):
        if os.getpid() != self._pid:
            return  # Never touch inherited driver leases/threads after fork.
        with self._cv:
            if self._closing:
                self._cv.wait_for(lambda: not self._pointer.value)
                return
            self._closing = True
            self._cv.wait_for(lambda: self._active == 0)
            if self._pointer.value:
                self._check(self._lib.tfllm_torch_destroy(self._pointer))
                self._pointer = C.c_void_p()
            self._cv.notify_all()
        _contexts.pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lib"):
            self.close()

    def __getstate__(self):
        raise TypeError("NPU runtime handles are process-local; save module.state_dict(), not the runtime")


def _linear_impl(x: torch.Tensor, runtime: int, weight: int, out_features: int) -> torch.Tensor:
    x = _tensor(x)
    if x.ndim < 1 or x.shape[-1] <= 0 or out_features <= 0:
        raise ValueError("invalid Linear shape")
    y = torch.empty((*x.shape[:-1], out_features), dtype=x.dtype, device="cpu")
    context = _context(runtime)
    op = "linear_f32" if x.dtype == torch.float32 else "linear"
    if x.dtype == torch.float32: _external_dtype(context)
    context._call(op, weight, x.data_ptr(), y.data_ptr(), x.numel() // x.shape[-1], x.shape[-1], out_features)
    return y


# Keep a dispatcher operator (including fake/compile support), but register the
# fixed, fresh-output kernel directly. custom_op's generic per-call wrapper and
# alias checks are measurable for small synchronous projections on the board.
_linear_library = torch.library.Library("tfllm", "FRAGMENT")
_linear_library.define(
    "linear(Tensor x, SymInt runtime, SymInt weight, SymInt out_features) -> Tensor",
    tags=(torch.Tag.pt2_compliant_tag,),
)
_linear_library.impl("linear", _linear_impl, "CompositeExplicitAutograd")
_linear = torch.ops.tfllm.linear.default


def _linear_backward(ctx, grad):
    raise RuntimeError("TFLLM is inference-only; use torch.inference_mode()")


torch.library.register_autograd(_linear, _linear_backward)


@torch.library.register_fake("tfllm::linear")
def _linear_fake(x, runtime, weight, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


@torch.library.custom_op("tfllm::linear_fp32", mutates_args=())
def _linear_fp32(x: torch.Tensor, runtime: int, weight: int, out_features: int) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise TypeError("FP32 Linear bridge expects CPU float32")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("TFLLM is inference-only; use torch.inference_mode()")
    if x.ndim < 1 or x.shape[-1] <= 0 or out_features <= 0:
        raise ValueError("invalid Linear shape")
    x = x.contiguous()
    y = x.new_empty((*x.shape[:-1], out_features))
    context = _context(runtime)
    _external_dtype(context)
    context._call("linear_f32", weight, x.data_ptr(), y.data_ptr(), x.numel() // x.shape[-1], x.shape[-1], out_features)
    return y


@_linear_fp32.register_fake
def _linear_fp32_fake(x, runtime, weight, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


@torch.library.custom_op("tfllm::pointwise", mutates_args=())
def _pointwise(x: torch.Tensor, runtime: int, weight: int, channels: int,
               stride: int, pad_h: int, pad_w: int) -> torch.Tensor:
    x = _tensor(x)
    if x.dtype == torch.float32:
        return _pointwise_fp32(x, runtime, weight, channels, stride, pad_h, pad_w)
    if (x.ndim != 4 or min(x.shape) <= 0 or channels <= 0 or not 1 <= stride <= 7
            or not 0 <= pad_h <= 7 or not 0 <= pad_w <= 7):
        raise ValueError("invalid pointwise shape/geometry")
    oh, ow = (x.shape[2]+2*pad_h-1)//stride+1, (x.shape[3]+2*pad_w-1)//stride+1
    context = _context(runtime)
    if not hasattr(context._lib, "tfllm_torch_pointwise"):
        raise RuntimeError("rebuild TFLLM with native pointwise support")
    y = x.new_empty((x.shape[0], channels, oh, ow))
    shape = (C.c_size_t * 8)(*x.shape, channels, stride, pad_h, pad_w)
    context._call("pointwise", weight, x.data_ptr(), y.data_ptr(), shape)
    return y


@_pointwise.register_fake
def _pointwise_fake(x, runtime, weight, channels, stride, pad_h, pad_w):
    oh, ow = (x.shape[2]+2*pad_h-1)//stride+1, (x.shape[3]+2*pad_w-1)//stride+1
    return x.new_empty((x.shape[0], channels, oh, ow))


@torch.library.custom_op("tfllm::pointwise_fp32", mutates_args=())
def _pointwise_fp32(x: torch.Tensor, runtime: int, weight: int, channels: int,
                    stride: int, pad_h: int, pad_w: int) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.float32:
        raise TypeError("FP32 pointwise expects CPU float32")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("TFLLM is inference-only; use torch.inference_mode()")
    if (x.ndim != 4 or min(x.shape) <= 0 or channels <= 0 or not 1 <= stride <= 7
            or not 0 <= pad_h <= 7 or not 0 <= pad_w <= 7):
        raise ValueError("invalid FP32 pointwise shape/geometry")
    x = x.contiguous()
    oh, ow = (x.shape[2]+2*pad_h-1)//stride+1, (x.shape[3]+2*pad_w-1)//stride+1
    context = _context(runtime)
    _external_dtype(context)
    if not hasattr(context._lib, "tfllm_torch_pointwise_fp32"):
        raise RuntimeError("rebuild TFLLM with native FP32 pointwise support")
    y = x.new_empty((x.shape[0], channels, oh, ow))
    shape = (C.c_size_t * 8)(*x.shape, channels, stride, pad_h, pad_w)
    context._call("pointwise_fp32", weight, x.data_ptr(), y.data_ptr(), shape)
    return y


@_pointwise_fp32.register_fake
def _pointwise_fp32_fake(x, runtime, weight, channels, stride, pad_h, pad_w):
    return _pointwise_fake(x, runtime, weight, channels, stride, pad_h, pad_w)


@torch.library.custom_op("tfllm::conv2d", mutates_args=())
def _conv(x: torch.Tensor, runtime: int, weight: int, channels: int, height: int, width: int) -> torch.Tensor:
    return _native_conv(x, runtime, weight, channels, height, width)


def _native_conv(x, runtime, weight, channels, height, width):
    x = _tensor(x)
    if x.ndim != 4 or min(x.shape) <= 0 or min(channels, height, width) <= 0:
        raise ValueError("invalid Conv2d shape")
    y = torch.empty((x.shape[0], channels, height, width), dtype=x.dtype, device="cpu")
    context = _context(runtime)
    op = "conv_f32" if x.dtype == torch.float32 else "conv"
    if x.dtype == torch.float32: _external_dtype(context)
    context._call(op, weight, x.data_ptr(), y.data_ptr(), x.shape[0], x.shape[2], x.shape[3], x.shape[1], channels, height, width)
    return y


@_conv.register_fake
def _conv_fake(x, runtime, weight, channels, height, width):
    return x.new_empty((x.shape[0], channels, height, width))


@torch.library.custom_op("tfllm::conv_transpose2d", mutates_args=())
def _conv_transpose(x: torch.Tensor, runtime: int, weight: int, channels: int, height: int, width: int) -> torch.Tensor:
    return _native_conv(x, runtime, weight, channels, height, width)


@_conv_transpose.register_fake
def _conv_transpose_fake(x, runtime, weight, channels, height, width):
    return x.new_empty((x.shape[0], channels, height, width))


def _strided_attention_available(q,k,v,shape,context):
    symbol="tfllm_torch_attention_strided_f32" if q.dtype==torch.float32 else "tfllm_torch_attention_strided_f16"
    return (shape[1]==shape[2] and (shape[3]>1024 or shape[4]>1024)
            and all(x.stride(-1)==1 and x.stride(-2)>=x.shape[-1] for x in (q,k,v))
            and (context is None or hasattr(context._lib,symbol)))


@torch.library.custom_op("tfllm::attention", mutates_args=())
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, runtime: int,
               is_causal: bool, scale: float, enable_gqa: bool) -> torch.Tensor:
    q, k, v = _attention_tensors(query, key, value, contiguous=False)
    if q.dtype == torch.float32:return _attention_fp32(q, k, v, runtime, is_causal, scale, enable_gqa)
    shape = _attention_geometry(q, k, v, enable_gqa)
    if max(shape[4:]) > 16384:
        raise ValueError("fused attention keys/channels must be <= 16384")
    if not _fused_scale(scale):
        raise ValueError("fused attention scale must be finite and positive in float32")
    context = _context(runtime)
    if context.attention_fast and max(shape[5:])>256:raise ValueError("FAST attention head dimensions must be <=256")
    if _strided_attention_available(q,k,v,shape,context):
        y=torch.empty((shape[0],shape[3],shape[1],shape[6]),dtype=q.dtype).transpose(1,2)
        params=(C.c_size_t*7)(*shape)
        strides=(C.c_size_t*12)(*(s for x in (q,k,v,y) for s in x.stride()[:3]))
        context._call("attention_strided_f16",q.data_ptr(),k.data_ptr(),v.data_ptr(),y.data_ptr(),params,strides,int(is_causal),scale)
        return y
    y = q.new_empty((*q.shape[:-1], v.shape[-1]))
    params = (C.c_size_t * 7)(*shape)
    eligible = (shape[0] * shape[1] >= 32 and shape[1] == shape[2] and shape[3] <= 1024
                and shape[4] <= 1024 and shape[5] == shape[6] and shape[5] <= 256)
    if eligible and not is_causal and all(x.stride(-1) == 1 for x in (q,k,v)) and hasattr(context._lib, "tfllm_torch_window_f16"):
        strides = (C.c_size_t * 9)(*(s for x in (q,k,v) for s in x.stride()[:3]))
        if context._call("window_f16", q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
                         params, strides, scale, allow_unsupported=True):
            return y
    q, k, v = _attention_tensors(q,k,v)
    context._call("attention", q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
                            params, int(is_causal), scale)
    return y


@_attention.register_fake
def _attention_fake(query, key, value, runtime, is_causal, scale, enable_gqa):
    shape=_attention_geometry(query,key,value,enable_gqa)
    if _strided_attention_available(query,key,value,shape,_contexts.get(runtime)):
        return query.new_empty((shape[0],shape[3],shape[1],shape[6])).transpose(1,2)
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("tfllm::attention_fp32", mutates_args=())
def _attention_fp32(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, runtime: int,
                    is_causal: bool, scale: float, enable_gqa: bool) -> torch.Tensor:
    if any(x.device.type != "cpu" or x.dtype != torch.float32 for x in (query, key, value)):
        raise TypeError("FP32 attention bridge expects CPU float32 Q/K/V")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value)):
        raise RuntimeError("TFLLM is inference-only; use torch.inference_mode()")
    shape = _attention_geometry(query, key, value, enable_gqa)
    if max(shape[4:]) > 16384:
        raise ValueError("fused attention keys/channels must be <= 16384")
    if not _fused_scale(scale):
        raise ValueError("fused attention scale must be finite and positive in float32")
    context = _context(runtime)
    _external_dtype(context)
    if context.attention_fast and max(shape[5:])>256:raise ValueError("FAST attention head dimensions must be <=256")
    if _strided_attention_available(query,key,value,shape,context):
        y=torch.empty((shape[0],shape[3],shape[1],shape[6]),dtype=query.dtype).transpose(1,2)
        params=(C.c_size_t*7)(*shape)
        strides=(C.c_size_t*12)(*(s for x in (query,key,value,y) for s in x.stride()[:3]))
        context._call("attention_strided_f32",query.data_ptr(),key.data_ptr(),value.data_ptr(),y.data_ptr(),params,strides,int(is_causal),scale)
        return y
    eligible = (shape[0] * shape[1] >= 32 and shape[1] == shape[2] and shape[3] <= 1024
                and shape[4] <= 1024 and shape[5] == shape[6] and shape[5] <= 256)
    if eligible and not is_causal and all(x.stride(-1) == 1 for x in (query, key, value)) and hasattr(context._lib, "tfllm_torch_window_f32"):
        y = query.new_empty((*query.shape[:-1], value.shape[-1]))
        params = (C.c_size_t * 7)(*shape)
        strides = (C.c_size_t * 9)(*(s for x in (query, key, value) for s in x.stride()[:3]))
        if context._call("window_f32", query.data_ptr(), key.data_ptr(), value.data_ptr(),
                         y.data_ptr(), params, strides, scale, allow_unsupported=True):
            return y
    q, k, v = _attention_tensors(query, key, value)
    y = q.new_empty((*q.shape[:-1], v.shape[-1]))
    params = (C.c_size_t * 7)(*shape)
    context._call("attention_f32", q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(), params, int(is_causal), scale)
    return y


@_attention_fp32.register_fake
def _attention_fp32_fake(query, key, value, runtime, is_causal, scale, enable_gqa):
    shape=_attention_geometry(query,key,value,enable_gqa)
    if _strided_attention_available(query,key,value,shape,_contexts.get(runtime)):
        return query.new_empty((shape[0],shape[3],shape[1],shape[6])).transpose(1,2)
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("tfllm::attention_intervals", mutates_args=())
def _attention_intervals(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         intervals: torch.Tensor, runtime: int, scale: float, enable_gqa: bool) -> torch.Tensor:
    q, k, v = _attention_tensors(query, key, value)
    shape = _attention_geometry(q, k, v, enable_gqa)
    if max(shape[4:]) > 16384 or not _fused_scale(scale):
        raise ValueError("unsupported fused attention shape/scale")
    if intervals.device.type != "cpu" or intervals.dtype != torch.int64 or intervals.shape != (shape[0], shape[3], 2):
        raise ValueError("intervals must be CPU int64 [B,L,2]")
    intervals = intervals.contiguous()
    y = q.new_empty((*q.shape[:-1], v.shape[-1]))
    params = (C.c_size_t * 7)(*shape)
    context = _context(runtime)
    op = "attention_intervals_f32" if q.dtype == torch.float32 else "attention_intervals"
    if q.dtype == torch.float32: _external_dtype(context)
    context._call(op, q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
                            params, intervals.data_ptr(), scale)
    return y


@_attention_intervals.register_fake
def _attention_intervals_fake(query, key, value, intervals, runtime, scale, enable_gqa):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("tfllm::attention_blocks", mutates_args=())
def _attention_blocks(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      row_offsets: torch.Tensor, column_indices: torch.Tensor, runtime: int,
                      query_block_size: int, kv_block_size: int, scale: float, enable_gqa: bool) -> torch.Tensor:
    q, k, v = _attention_tensors(query, key, value)
    shape = _attention_geometry(q, k, v, enable_gqa)
    if max(shape[5:]) > 16384 or not _fused_scale(scale):
        raise ValueError("unsupported block attention channel count/scale")
    if not 1 <= query_block_size <= 1024 or not 1 <= kv_block_size <= 16384:
        raise ValueError("block attention requires query block 1..1024 and KV block 1..16384")
    for tensor in (row_offsets, column_indices):
        if tensor.device.type != 'cpu' or tensor.dtype != torch.int64 or tensor.ndim != 1:
            raise ValueError("block CSR offsets/indices must be one-dimensional CPU int64 tensors")
    qblocks = (shape[3] + query_block_size - 1) // query_block_size
    if row_offsets.numel() != shape[0] * shape[1] * qblocks + 1:
        raise ValueError("block CSR row_offsets length must be B*Hq*ceil(L/query_block_size)+1")
    offsets, indices = row_offsets.contiguous(), column_indices.contiguous()
    y = q.new_empty((*q.shape[:-1], v.shape[-1]))
    params = (C.c_size_t * 7)(*shape)
    context = _context(runtime)
    op = "attention_blocks_f32" if q.dtype == torch.float32 else "attention_blocks"
    if q.dtype == torch.float32: _external_dtype(context)
    context._call(op, q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
        params, offsets.data_ptr(), offsets.numel(), indices.data_ptr(), indices.numel(),
        query_block_size, kv_block_size, scale)
    return y


@_attention_blocks.register_fake
def _attention_blocks_fake(query, key, value, row_offsets, column_indices, runtime,
                           query_block_size, kv_block_size, scale, enable_gqa):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("tfllm::attention_mask", mutates_args=())
def _attention_mask(query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask:torch.Tensor,
                    runtime:int,is_causal:bool,scale:float,enable_gqa:bool)->torch.Tensor:
    q,k,v=_attention_tensors(query,key,value)
    shape=_attention_geometry(q,k,v,enable_gqa);context=_context(runtime)
    if not context.attention_fast or max(shape[5:])>256 or shape[4]>16384 or not _fused_scale(scale):
        raise ValueError("FAST masked attention requires head dimensions <=256, keys<=16384 and a positive finite scale")
    if mask.device.type!="cpu" or mask.requires_grad or mask.dtype not in (torch.bool,torch.float16,torch.float32,torch.bfloat16):
        raise ValueError("attention mask must be a CPU bool/floating inference tensor")
    if mask.dtype!=torch.bool:
        if bool((torch.isnan(mask)|torch.isposinf(mask)).any()):raise ValueError("attention bias must be finite or -inf")
        mask=mask.float()
    mask=torch.broadcast_to(mask,(shape[0],shape[1],shape[3],shape[4]))
    strides=(C.c_size_t*4)(*mask.stride());params=(C.c_size_t*7)(*shape)
    out=q.new_empty((*q.shape[:-1],v.shape[-1]))
    context._call("attention_mask_f32" if q.dtype==torch.float32 else "attention_mask_f16",
        q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),params,mask.data_ptr(),strides,
        int(mask.dtype==torch.bool),int(is_causal),scale)
    return out


@_attention_mask.register_fake
def _attention_mask_fake(query,key,value,mask,runtime,is_causal,scale,enable_gqa):
    return query.new_empty((*query.shape[:-1],value.shape[-1]))
