"""Independent NPU collectors over a shared, read-only calibration file."""

import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

_WORKER_LAUNCHER = """
import json, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[1])
from TFConvertor.quantization.parallel_collect import worker
worker(json.loads(Path(sys.argv[2]).read_text()))
"""


def merge_ranges(parts, expected):
    """MinMax statistics combine by minimum/maximum, never by averaging."""
    expected = set(expected)
    result = {}
    if not parts:
        raise ValueError("no worker ranges")
    for ranges in parts:
        if set(ranges) != expected:
            raise ValueError("worker tensor names differ from the source graph")
        for name, values in ranges.items():
            if len(values) != 2:
                raise ValueError("invalid worker range for " + name)
            low, high = map(float, values)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("invalid worker range for " + name)
            old_low, old_high = result.get(name, (low, high))
            result[name] = min(old_low, low), max(old_high, high)
    return result


def merge_coverage(reports):
    entries, calls = {}, {}
    for report in reports:
        coverage = report["coverage"]
        for row in coverage["entries"]:
            key = tuple(row.get(k) for k in ("path", "operator", "status", "reason"))
            entries[key] = row
        for row in coverage["calls"]:
            key = tuple(row.get(k) for k in ("path", "status", "reason"))
            if key not in calls:
                calls[key] = dict(row, count=0)
            calls[key]["count"] += row["count"]
    return dict(
        scope=reports[0]["coverage"]["scope"],
        entries=list(entries.values()),
        calls=list(calls.values()),
    )


def _stop(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def collect_parallel(
    graph, images, threads, spatial, chip, workers, workdir=None, images_path=None
):
    started = time.perf_counter()
    requested = workers
    workers = min(workers, len(images))
    if workers < 1:
        raise ValueError("real calibration samples are required")
    temporary = (
        tempfile.TemporaryDirectory(prefix="tfdl-collect-") if workdir is None else None
    )
    directory = Path(temporary.name if temporary else workdir) / "collector-workers"
    directory.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    try:
        # No TFContext, executor, driver handle or live Torch module crosses a
        # process boundary. Parameters are copied before SDK Quantize mutates them.
        snapshot = SimpleNamespace(
            calls=graph.calls,
            params=graph.params,
            meta=graph.meta,
            outputs=graph.outputs,
        )
        payload = directory / "graph.pkl"
        with payload.open("wb") as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if images_path is None:
            images_path = directory / "images.f32"
            images.tofile(images_path)
        images_path = Path(images_path).resolve()
        if images_path.stat().st_size != images.nbytes:
            raise ValueError("calibration file size does not match the sample array")
        configs = []
        environment = dict(
            os.environ,
            OMP_NUM_THREADS=str(threads),
            OPENBLAS_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
        )
        # The example entry points add ConvertTools/python to sys.path rather
        # than PYTHONPATH. Reuse the caller's resolved imports in fresh workers.
        import_paths = [
            str(Path(path or ".").resolve())
            for path in sys.path
            if isinstance(path, str)
        ]
        for index in range(workers):
            config = dict(
                index=index,
                workers=workers,
                threads=threads,
                spatial=spatial,
                chip=chip,
                payload=str(payload.resolve()),
                images=str(images_path),
                shape=list(images.shape),
                output=str((directory / f"worker-{index}.json").resolve()),
            )
            path = directory / f"worker-{index}-config.json"
            path.write_text(json.dumps(config))
            configs.append(config)
            log = (directory / f"worker-{index}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _WORKER_LAUNCHER,
                        json.dumps(import_paths),
                        str(path.resolve()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            )
        pending = set(range(workers))
        while pending:
            for index in list(pending):
                status = processes[index].poll()
                if status is None:
                    continue
                if status:
                    tail = (directory / f"worker-{index}.log").read_text(
                        errors="replace"
                    )[-4000:]
                    raise RuntimeError(
                        f"NPU calibration worker {index} failed (exit {status}):\n{tail}"
                    )
                pending.remove(index)
                print(f"tfllm-npu: worker {index+1}/{workers} completed", flush=True)
            if pending:
                time.sleep(0.05)
        parts = [json.loads(Path(c["output"]).read_text()) for c in configs]
        reports = []
        for config, part in zip(configs, parts):
            expected_indices = list(range(config["index"], len(images), workers))
            if part["indices"] != expected_indices or part["report"]["samples"] != len(
                expected_indices
            ):
                raise ValueError("worker did not collect its complete sample shard")
            reports.append(part["report"])
        ranges = merge_ranges([p["ranges"] for p in parts], graph.meta)
        versions = {(r["torch_version"], r["runtime_library_sha256"]) for r in reports}
        if len(versions) != 1:
            raise ValueError(
                "collector workers used different Torch/NPU runtime versions"
            )
        coverage = merge_coverage(reports)
        report = dict(
            backend="tfllm-npu",
            device="cpu",
            approximate=True,
            samples=len(images),
            tensors=len(ranges),
            workers=workers,
            workers_requested=requested,
            threads=threads,
            seconds=time.perf_counter() - started,
            torch_version=reports[0]["torch_version"],
            runtime_library_sha256=reports[0]["runtime_library_sha256"],
            coverage=coverage,
            merge="per-tensor minimum of minima / maximum of maxima",
            start_method="fresh-subprocess",
            worker_reports=[
                dict(
                    index=i,
                    indices=part["indices"],
                    pid=part["pid"],
                    wall_seconds=part["wall_seconds"],
                    collector=part["report"],
                )
                for i, part in enumerate(parts)
            ],
        )
        return ranges, report
    finally:
        _stop(processes)
        for log in logs:
            log.close()
        if temporary:
            temporary.cleanup()


def worker(config):
    def terminate(signum, frame):
        # Let collect's finally block release its runtime after an in-flight
        # synchronous native call returns. The parent can still kill a hang.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    import numpy as np
    from .torch_backend import collect

    start = time.perf_counter()
    with Path(config["payload"]).open("rb") as stream:
        graph = pickle.load(stream)
    images = np.memmap(
        config["images"], dtype=np.float32, mode="r", shape=tuple(config["shape"])
    )
    indices = list(range(config["index"], len(images), config["workers"]))
    ranges, report = collect(
        graph,
        images[config["index"] :: config["workers"]],
        "tfllm-npu",
        config["threads"],
        config["spatial"],
        config["chip"],
    )
    output = Path(config["output"])
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            dict(
                ranges=ranges,
                report=report,
                indices=indices,
                pid=os.getpid(),
                wall_seconds=time.perf_counter() - start,
            )
        )
    )
    temporary.replace(output)
