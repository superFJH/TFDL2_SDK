from .runtime import NpuRuntime
from .modules import NpuLinear, NpuConv2d, NpuAttention, replace_linear
from .optimize import optimize, restore, optimization_report, Unsupported
from .compiler import make_backend
from .sparse_attention import BlockSparseMask, NpuBlockSparseAttention

__all__ = ["NpuRuntime", "NpuLinear", "NpuConv2d", "NpuAttention", "replace_linear",
           "optimize", "restore", "optimization_report", "Unsupported", "make_backend",
           "BlockSparseMask", "NpuBlockSparseAttention"]
