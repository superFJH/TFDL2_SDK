#!/usr/bin/env python3
"""Minimal regression test for native UINT8 tensor-by-tensor Op.Mul.

The selected qinfo makes the real-valued right input exactly 1.0 and the
output quantization identical to the left input, so the expected result is an
exact copy of the left UINT8 codes.  No model weights or custom op are used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from TFDL2 import Op, TFContext, TFExecutor
from TFDL2.Common import TFDataType


def run_case(width: int, frugal_mode: bool) -> dict[str, object]:
    if width < 1 or width > 256:
        raise ValueError("width must be in [1, 256]")

    shape = (1, 1, 1, width)
    context = TFContext(f"Uint8TensorMulW{width}F{int(frugal_mode)}")
    with context:
        left = Op.Placeholder2(context, shape, TFDataType.TFDL_UINT8)
        right = Op.Placeholder2(context, shape, TFDataType.TFDL_UINT8)
        output = Op.Mul(left, right)

        # left:  real = code * 1.0
        # right: real = code * 0.5; code 2 therefore represents exactly 1.0
        # output: real = code * 1.0
        # Hence output_code = left_code, with an exact requant multiplier 0.5.
        ranges = (
            (str(left), 255.0, 0.0),
            (str(right), 127.5, 0.0),
            (str(output), 255.0, 0.0),
        )
        for name, high, low in ranges:
            if not context.AddInt8Config(name, high, low):
                raise RuntimeError(f"AddInt8Config failed for {name}")
        context.SetOutputs([str(output)])

    executor = TFExecutor(
        context,
        {"UseHardware": False, "FrugalMode": frugal_mode},
    )
    left_codes = np.arange(width, dtype=np.uint8).reshape(shape)
    right_codes = np.full(shape, 2, dtype=np.uint8)
    for tensor, value in zip(
        executor.GetInputs(), (left_codes, right_codes), strict=True
    ):
        tensor.fromNumpy(np.ascontiguousarray(value))

    actual = executor()[0].toNumpy().reshape(-1).astype(np.uint8)
    expected = left_codes.reshape(-1)
    return {
        "width": width,
        "frugal_mode": frugal_mode,
        "use_hardware": False,
        "qinfo": {
            "left": {"min": 0.0, "max": 255.0, "scale": 1.0, "zero": 0},
            "right": {"min": 0.0, "max": 127.5, "scale": 0.5, "zero": 0},
            "output": {"min": 0.0, "max": 255.0, "scale": 1.0, "zero": 0},
        },
        "left_codes": expected.tolist(),
        "right_codes": right_codes.reshape(-1).tolist(),
        "expected_codes": expected.tolist(),
        "actual_codes": actual.tolist(),
        "bit_exact": bool(np.array_equal(actual, expected)),
        "mismatch_indices": np.flatnonzero(actual != expected).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=int,
        default=8,
        help="flattened element count; 8 is the minimal vector-path repro",
    )
    parser.add_argument(
        "--require-fixed",
        action="store_true",
        help="exit non-zero while either executor mode is not bit-exact",
    )
    args = parser.parse_args()

    results = [run_case(args.width, frugal) for frugal in (False, True)]
    print(json.dumps({"cases": results}, indent=2))
    if args.require_fixed and not all(case["bit_exact"] for case in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
