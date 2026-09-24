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


def _tensor(x):
    if x.device.type != "cpu" or x.dtype != torch.float16:
        raise TypeError("TFLLM expects CPU float16 tensors; convert explicitly before inference")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("TFLLM is inference-only; use torch.inference_mode()")
    return x.contiguous()


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
    if hasattr(lib, "tfllm_torch_attention_intervals"):
        lib.tfllm_torch_attention_intervals.restype = C.c_int
        lib.tfllm_torch_attention_intervals.argtypes = [C.c_void_p] * 5 + [C.POINTER(C.c_size_t), C.c_void_p, C.c_float]
    if hasattr(lib, "tfllm_torch_attention_blocks"):
        lib.tfllm_torch_attention_blocks.restype = C.c_int
        lib.tfllm_torch_attention_blocks.argtypes = ([C.c_void_p] * 5 + [C.POINTER(C.c_size_t),
            C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_float])
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
    """One shared runtime per chip. CPU mode is an explicit development oracle."""
    def __init__(self, chip=0, *, backend="npu", memory_fraction=0.8, library=None,
                 attention_fuse_workers=16, attention_score_mib=4,
                 attention_pipeline=True, attention_lease_tiles=4):
        if backend not in ("npu", "cpu"):
            raise ValueError("backend must be npu or cpu")
        for name, value, limit in (("attention_fuse_workers", attention_fuse_workers, 32),
                                   ("attention_score_mib", attention_score_mib, 16),
                                   ("attention_lease_tiles", attention_lease_tiles, 16)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be 1..{limit}")
        if type(attention_pipeline) is not bool:
            raise TypeError("attention_pipeline must be bool")
        self._cv = threading.Condition()
        self._pid = os.getpid()
        self._active, self._closing = 0, False
        self._pointer = C.c_void_p()
        self._lib = _library(library)
        self.backend, self.chip = backend, chip
        self.id = next(_ids)
        attention = (C.c_size_t * 4)(attention_fuse_workers, attention_score_mib,
                                    int(attention_pipeline), attention_lease_tiles)
        self._check(self._lib.tfllm_torch_create(int(backend == "npu"), chip, memory_fraction,
                                              attention, C.byref(self._pointer)))
        _contexts[self.id] = self

    def _check(self, status):
        if status:
            raise RuntimeError(self._lib.tfllm_torch_error().decode("utf-8", errors="replace"))

    def _call(self, name, *args):
        if os.getpid() != self._pid:
            raise RuntimeError("create a new NPU runtime after fork; handles cannot cross processes")
        with self._cv:
            if self._closing or not self._pointer.value:
                raise RuntimeError("TFLLM runtime is closed")
            self._active += 1
        try:
            self._check(getattr(self._lib, "tfllm_torch_" + name)(self._pointer, *args))
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
        handle = C.c_uint64()
        if spatial is None:
            if weight.ndim != 2 or min(weight.shape) <= 0:
                raise ValueError("matrix weight must be nonempty [out_features, in_features]")
            self._call("weight", weight.data_ptr(), *weight.shape, int(dynamic), C.byref(handle))
        else:
            if weight.ndim != 4 or min(weight.shape) <= 0 or len(spatial) != 5 or min(spatial) < 0:
                raise ValueError("spatial weights require OIHW and stride/top/bottom/left/right")
            params = (C.c_size_t * 9)(*weight.shape, *spatial)
            self._call("conv_weight", weight.data_ptr(), params, C.byref(handle))
        return PackedWeight(self, handle.value)

    def linear(self, x, weight, out_features):
        if weight.runtime is not self:
            raise ValueError("weight belongs to a different runtime")
        return _linear(_tensor(x), self.id, weight.handle, out_features)

    def conv(self, x, weight, out_channels, height, width):
        if weight.runtime is not self:
            raise ValueError("weight belongs to a different runtime")
        return _conv(_tensor(x), self.id, weight.handle, out_channels, height, width)

    def attention(self, query, key, value, *, is_causal=False, scale=None, enable_gqa=False):
        """Strict fused SDPA: no Torch score/probability tensors or fallback."""
        if self.backend != "npu":
            raise NotImplementedError("fused attention requires backend='npu'; CPU oracle uses implementation='torch'")
        q, k, v = map(_tensor, (query, key, value))
        shape = _attention_geometry(q, k, v, enable_gqa)
        scale = shape[5] ** -0.5 if scale is None else float(scale)
        return _attention(q, k, v, self.id, is_causal, scale, enable_gqa)

    def attention_intervals(self, query, key, value, intervals, *, scale=None, enable_gqa=False):
        """Compact [B,L,2] int64 visible key intervals; empty rows return zero."""
        if self.backend != "npu":
            raise NotImplementedError("fused attention requires backend='npu'")
        if not hasattr(self._lib, "tfllm_torch_attention_intervals"):
            raise RuntimeError("rebuild libtfllm-torch for interval attention support")
        q, k, v = map(_tensor, (query, key, value))
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
        q, k, v = map(_tensor, (query, key, value))
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


@torch.library.custom_op("tfllm::linear", mutates_args=())
def _linear(x: torch.Tensor, runtime: int, weight: int, out_features: int) -> torch.Tensor:
    x = _tensor(x)
    if x.ndim < 1 or x.shape[-1] <= 0 or out_features <= 0:
        raise ValueError("invalid Linear shape")
    y = torch.empty((*x.shape[:-1], out_features), dtype=x.dtype, device="cpu")
    _context(runtime)._call("linear", weight, x.data_ptr(), y.data_ptr(), x.numel() // x.shape[-1], x.shape[-1], out_features)
    return y


@_linear.register_fake
def _linear_fake(x, runtime, weight, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


@torch.library.custom_op("tfllm::conv2d", mutates_args=())
def _conv(x: torch.Tensor, runtime: int, weight: int, channels: int, height: int, width: int) -> torch.Tensor:
    x = _tensor(x)
    if x.ndim != 4 or min(x.shape) <= 0 or min(channels, height, width) <= 0:
        raise ValueError("invalid Conv2d shape")
    y = torch.empty((x.shape[0], channels, height, width), dtype=x.dtype, device="cpu")
    _context(runtime)._call("conv", weight, x.data_ptr(), y.data_ptr(), x.shape[0], x.shape[2], x.shape[3], x.shape[1], channels, height, width)
    return y


@_conv.register_fake
def _conv_fake(x, runtime, weight, channels, height, width):
    return x.new_empty((x.shape[0], channels, height, width))


@torch.library.custom_op("tfllm::attention", mutates_args=())
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, runtime: int,
               is_causal: bool, scale: float, enable_gqa: bool) -> torch.Tensor:
    q, k, v = map(_tensor, (query, key, value))
    shape = _attention_geometry(q, k, v, enable_gqa)
    if max(shape[4:]) > 16384:
        raise ValueError("fused attention keys/channels must be <= 16384")
    if not _fused_scale(scale):
        raise ValueError("fused attention scale must be finite and positive in float32")
    y = q.new_empty((*q.shape[:-1], v.shape[-1]))
    params = (C.c_size_t * 7)(*shape)
    _context(runtime)._call("attention", q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
                            params, int(is_causal), scale)
    return y


@_attention.register_fake
def _attention_fake(query, key, value, runtime, is_causal, scale, enable_gqa):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("tfllm::attention_intervals", mutates_args=())
def _attention_intervals(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         intervals: torch.Tensor, runtime: int, scale: float, enable_gqa: bool) -> torch.Tensor:
    q, k, v = map(_tensor, (query, key, value))
    shape = _attention_geometry(q, k, v, enable_gqa)
    if max(shape[4:]) > 16384 or not _fused_scale(scale):
        raise ValueError("unsupported fused attention shape/scale")
    if intervals.device.type != "cpu" or intervals.dtype != torch.int64 or intervals.shape != (shape[0], shape[3], 2):
        raise ValueError("intervals must be CPU int64 [B,L,2]")
    intervals = intervals.contiguous()
    y = q.new_empty((*q.shape[:-1], v.shape[-1]))
    params = (C.c_size_t * 7)(*shape)
    _context(runtime)._call("attention_intervals", q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
                            params, intervals.data_ptr(), scale)
    return y


@_attention_intervals.register_fake
def _attention_intervals_fake(query, key, value, intervals, runtime, scale, enable_gqa):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@torch.library.custom_op("tfllm::attention_blocks", mutates_args=())
def _attention_blocks(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      row_offsets: torch.Tensor, column_indices: torch.Tensor, runtime: int,
                      query_block_size: int, kv_block_size: int, scale: float, enable_gqa: bool) -> torch.Tensor:
    q, k, v = map(_tensor, (query, key, value))
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
    _context(runtime)._call("attention_blocks", q.data_ptr(), k.data_ptr(), v.data_ptr(), y.data_ptr(),
        params, offsets.data_ptr(), offsets.numel(), indices.data_ptr(), indices.numel(),
        query_block_size, kv_block_size, scale)
    return y


@_attention_blocks.register_fake
def _attention_blocks_fake(query, key, value, row_offsets, column_indices, runtime,
                           query_block_size, kv_block_size, scale, enable_gqa):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))
