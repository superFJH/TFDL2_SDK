"""One request thread per shard, one shared read-only model and NPU runtime."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn

from .graph import sha256
from .parallel_collect import merge_ranges


def _bounds(value):
    return torch.stack(
        (value.amin(), value.amax()) if value.dtype == torch.bool else torch.aminmax(value)
    ).to(torch.float64)


class RangeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.names = [name for call in model.graph.calls for name in call["outputs"]]
        self.eval()

    def forward(self, image):
        bounds = []
        outputs = self.model(image, lambda name, value: bounds.append(_bounds(value)))
        return outputs, torch.stack(bounds)


def collect_threaded(graph, images, threads, spatial, chip, workers, workdir, compile_model):
    from .torch_backend import TorchGraph
    from tfllm.torch import NpuRuntime, optimize, optimization_report, restore

    workers_requested = workers
    workers = min(workers, len(images))
    if workers < 1:
        raise ValueError("real calibration samples are required")
    previous_threads = torch.get_num_threads()
    runtime = model = snapshot = None
    stop = threading.Event()
    directory = None
    if workdir is not None:
        directory = Path(workdir) / "collector-workers"
        directory.mkdir(parents=True, exist_ok=False)
    reports = [None] * workers
    try:
        # Torch's setting is process-wide. Set/restore it once on the owner,
        # never race set_num_threads() calls from individual request threads.
        torch.set_num_threads(threads)
        model = TorchGraph(graph)
        runtime = NpuRuntime(chip=chip, backend="npu", memory_fraction=0.125,
                             spatial_convolution=spatial)
        optimize(model, runtime=runtime, strict=False,
                 operators=("conv2d", "linear"), precision="auto")
        names = [name for call in graph.calls for name in call["outputs"]]
        if len(names) != len(set(names)) or set(names) != set(graph.meta):
            raise ValueError("threaded collector requires unique recorded tensor names")
        if compile_model:
            from tfllm.torch import compile_snapshot

            with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                snapshot = compile_snapshot(
                    RangeModel(model), torch.from_numpy(images[0:1].copy())
                )

        def shard(index):
            indices = list(range(index, len(images), workers))
            started = time.perf_counter()
            total = None
            with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                for sample in indices:
                    if stop.is_set():
                        raise RuntimeError("calibration shard stopped after another worker failed")
                    image = torch.from_numpy(images[sample:sample+1].copy())
                    if snapshot is None:
                        observed = []
                        model(image, lambda name, value: observed.append(_bounds(value)))
                        current = torch.stack(observed)
                    else:
                        _, current = snapshot(image)
                    total = current if total is None else torch.stack(
                        (torch.minimum(total[:, 0], current[:, 0]),
                         torch.maximum(total[:, 1], current[:, 1])), dim=1
                    )
                    print(f"tfllm-npu: thread {index+1}/{workers} calibrated sample {sample+1}/{len(images)}", flush=True)
                bounds = total.numpy()
            if not np.isfinite(bounds).all():
                raise ValueError("nonfinite threaded calibration observation")
            ranges = {name: (float(lo), float(hi)) for name, (lo, hi) in zip(names, bounds)}
            part = dict(index=index, indices=indices, pid=os.getpid(),
                        thread_id=threading.get_ident(), samples=len(indices),
                        seconds=time.perf_counter()-started, ranges=ranges)
            if directory is not None:
                (directory / f"thread-{index}.json").write_text(json.dumps(part, indent=2))
            return part

        # A failed shard cancels remaining samples. Executor shutdown drains
        # all in-flight synchronous native calls before restore/runtime.close.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tfdl-calibration") as pool:
            futures = {pool.submit(shard, index): index for index in range(workers)}
            try:
                for future in as_completed(futures):
                    reports[futures[future]] = future.result()
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        ranges = merge_ranges([r["ranges"] for r in reports], graph.meta)
        coverage = optimization_report(model).to_dict()
        if not any(c["status"] == "npu" and c["count"] for c in coverage["calls"]):
            raise RuntimeError("no actual NPU calibration calls")
        report = dict(
            backend="tfllm-npu", device="cpu", approximate=True,
            samples=len(images), tensors=len(ranges), workers=workers,
            workers_requested=workers_requested, threads=threads, executor="threads",
            sharing="one process, read-only eval model and one runtime; private observations per request thread",
            torch_version=torch.__version__, runtime_library_sha256=sha256(Path(runtime._lib._name)),
            coverage=coverage, compiled_snapshot=None if snapshot is None else snapshot.report,
            coverage_scope="eager collection" if snapshot is None else "capture/warmup; compiled calls use captured TFLLM custom operators",
            worker_reports=[{k:v for k,v in part.items() if k != "ranges"} for part in reports],
        )
        return ranges, report
    finally:
        try:
            if snapshot is not None:
                snapshot.close()
            if model is not None and runtime is not None:
                try:
                    restore(model)
                finally:
                    runtime.close()
        finally:
            torch.set_num_threads(previous_threads)
