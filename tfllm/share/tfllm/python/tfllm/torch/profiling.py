"""Opt-in host profiling of Torch work and synchronous TFLLM calls."""
from pathlib import Path
import time

import torch


def _category(name):
    if name.startswith("tfllm::"):
        return "tfllm_call"
    if name in {"aten::to", "aten::_to_copy", "aten::copy_", "aten::clone",
                "aten::contiguous", "aten::reshape", "aten::_reshape_alias"}:
        return "conversion_layout"
    if "layer_norm" in name or "batch_norm" in name or "group_norm" in name:
        return "normalization"
    if any(word in name for word in ("gelu", "silu", "relu", "sigmoid")):
        return "activation"
    if any(word in name for word in ("scaled_dot_product", "softmax", "bmm", "matmul", "einsum")):
        return "torch_attention_matmul"
    if "conv" in name:
        return "torch_convolution"
    if "upsample" in name or "grid_sampler" in name:
        return "resize"
    return "other_torch"


class InferenceProfiler:
    """Profile exactly the block supplied by the caller, without replaying it.

    TFLLM custom-op spans include CPU quantization, cache maintenance, driver
    work, device waiting and dequantization. They are NOT device execution time.
    Self host times avoid counting parent/child spans twice, but concurrent
    threads may overlap; their sum is not an end-to-end latency decomposition.
    Warm up and measure ordinary latency separately, outside this context.
    """
    def __init__(self, *, trace_path=None, top=30):
        if type(top) is not int or top <= 0:
            raise ValueError("top must be a positive integer")
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.top = top
        self._used, self._report = False, None

    def __enter__(self):
        if self._used:
            raise RuntimeError("create a new InferenceProfiler for each measurement")
        self._used = True
        self._profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True)
        self._profiler.__enter__()
        self._start = time.perf_counter()
        return self

    def __exit__(self, kind, value, traceback):
        wall_ms = (time.perf_counter() - self._start) * 1000
        self._profiler.__exit__(kind, value, traceback)
        events = self._profiler.key_averages(group_by_input_shape=True)
        rows, categories = [], {}
        for event in events:
            # Ignore record_function annotations: they are useful in the trace
            # but are not Torch operations or native TFLLM calls.
            if not event.key.startswith(("aten::", "tfllm::")):
                continue
            category = _category(event.key)
            self_ms = event.self_cpu_time_total / 1000
            categories[category] = categories.get(category, 0.) + self_ms
            rows.append({"operator": event.key, "category": category,
                         "calls": event.count, "self_host_ms": self_ms,
                         "total_host_ms": event.cpu_time_total / 1000,
                         "input_shapes": event.input_shapes})
        rows.sort(key=lambda row: row["self_host_ms"], reverse=True)
        self._report = {
            "profiled_wall_ms": wall_ms, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "category_self_host_ms": categories,
            "top_operators": rows[:self.top], "operator_groups": len(rows),
            "note": "Host spans with profiler overhead; tfllm_call includes CPU work and NPU waiting. "
                    "Categories are operator self times, not an end-to-end latency decomposition. "
                    "Native CPU worker activity is inside the synchronous TFLLM call, not separately visible.",
        }
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler.export_chrome_trace(str(self.trace_path))
        return False

    def to_dict(self):
        if self._report is None:
            raise RuntimeError("read the profile after exiting its context")
        # Return independent containers so callers can annotate their report.
        import copy
        return copy.deepcopy(self._report)

    def __str__(self):
        report = self.to_dict()
        lines = [f"Profiled wall: {report['profiled_wall_ms']:.3f} ms; "
                 f"Torch threads: {report['torch_threads']}", report["note"],
                 "Self host ms / calls / operator / input shapes"]
        lines += [f"{row['self_host_ms']:10.3f} / {row['calls']:5d} / "
                  f"{row['operator']} / {row['input_shapes']}"
                  for row in report["top_operators"]]
        return "\n".join(lines)
