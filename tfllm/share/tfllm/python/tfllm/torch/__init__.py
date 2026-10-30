from .runtime import NpuRuntime
from .modules import NpuLinear, NpuConv2d, NpuConvTranspose2d, NpuAttention, replace_linear
from .optimize import optimize, restore, optimization_report, Unsupported
from .compiler import make_backend
from .sparse_attention import BlockSparseMask, NpuBlockSparseAttention
from .profiling import InferenceProfiler


def compile_snapshot(model, example_inputs, *, options=None):
    # Keep the optional compiler dependency out of ordinary eager imports.
    from .snapshot import compile_snapshot as create
    return create(model, example_inputs, options=options)

__all__ = ["NpuRuntime", "NpuLinear", "NpuConv2d", "NpuConvTranspose2d", "NpuAttention", "replace_linear",
           "optimize", "restore", "optimization_report", "Unsupported", "make_backend",
           "BlockSparseMask", "NpuBlockSparseAttention", "InferenceProfiler",
           "compile_snapshot"]
