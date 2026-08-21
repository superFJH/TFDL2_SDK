#!/usr/bin/env python3
"""Correctness, scaling, and latency checks for the ARM RMSNorm addon."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SDK_ROOT = THIS_DIR.parents[2]
PYTHON_DIR = SDK_ROOT / "Python"
BUILD_DIRS = sorted((PYTHON_DIR / "build").glob("lib.*"))
for path in reversed((*BUILD_DIRS, PYTHON_DIR)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_tfdl(addon: Path):
    from TFDL2 import Op, TFContext, TFExecutor
    from TFDL2.Common import TFDataType
    from TFDL2.utils import LoadCustomOp

    LoadCustomOp(str(addon))
    return Op, TFContext, TFExecutor, TFDataType


def _custom_output(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def rmsnorm_case(
    tfdl,
    shape: tuple[int, ...],
    dtype_name: str,
    threads: int,
    iterations: int,
    seed: int,
    eps: float,
) -> dict[str, object]:
    Op, TFContext, TFExecutor, TFDataType = tfdl
    numpy_dtype = np.float16 if dtype_name == "fp16" else np.float32
    tfdl_dtype = (
        TFDataType.TFDL_FLOAT16
        if dtype_name == "fp16"
        else TFDataType.TFDL_FLOAT
    )
    rng = np.random.default_rng(seed + sum(shape) + (dtype_name == "fp16"))
    source = rng.normal(scale=0.3, size=shape).astype(numpy_dtype)
    context = TFContext(
        f"ArmRMSNorm{dtype_name.upper()}D{shape[-1]}T{threads}"
    )
    with context:
        input_value = Op.Placeholder2(context, shape, tfdl_dtype)
        output = _custom_output(
            Op.Custom(
                (input_value,),
                (f"arm_rmsnorm_{dtype_name}_{shape[-1]}_{threads}",),
                "RMSNorm",
                json.dumps({"eps": eps, "threads": threads}),
            )
        )
    context.SetOutputs([str(output)])
    executor = TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )
    executor.GetInputs()[0].fromNumpy(np.ascontiguousarray(source))
    actual = executor()[0].toNumpy()
    source_float = source.astype(np.float32)
    reference = (
        source_float
        / np.sqrt(
            np.mean(source_float * source_float, axis=-1, keepdims=True)
            + eps
        )
    ).astype(numpy_dtype)
    difference = np.abs(
        actual.astype(np.float32) - reference.astype(np.float32)
    )
    tolerance = 2e-3 if dtype_name == "fp16" else 2e-5
    if float(difference.max()) > tolerance:
        raise AssertionError(
            f"RMSNorm {dtype_name} {shape}: max_abs={difference.max()} "
            f"> {tolerance}"
        )
    for _ in range(3):
        executor()
    started = time.perf_counter()
    for _ in range(iterations):
        executor()
    seconds = (time.perf_counter() - started) / iterations
    elements = int(np.prod(shape))
    return {
        "dtype": dtype_name,
        "shape": list(shape),
        "threads": threads,
        "elements": elements,
        "seconds": seconds,
        "effective_giga_elements_per_second": elements / seconds / 1e9,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--threads", type=int, action="append")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--addon-path",
        default=str(SDK_ROOT / "AddonOps/build/libTFDLAddOn.so"),
    )
    args = parser.parse_args()
    if min(args.sequence, args.hidden, args.heads, args.head_dim) <= 0:
        raise ValueError("RMSNorm dimensions must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    threads = args.threads or [0]
    if any(value < 0 for value in threads):
        raise ValueError("--threads values must be non-negative")
    shapes = (
        (1, args.sequence, args.hidden),
        (1, args.heads, args.sequence, args.head_dim),
    )
    tfdl = _load_tfdl(Path(args.addon_path).resolve())
    report = {
        "format": "mage-arm-rmsnorm-benchmark-v1",
        "cases": [],
    }
    for shape in shapes:
        for dtype_name in ("fp32", "fp16"):
            for thread_count in threads:
                item = rmsnorm_case(
                    tfdl,
                    shape,
                    dtype_name,
                    thread_count,
                    args.iterations,
                    args.seed,
                    args.eps,
                )
                report["cases"].append(item)
                print(json.dumps(item, sort_keys=True), flush=True)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
