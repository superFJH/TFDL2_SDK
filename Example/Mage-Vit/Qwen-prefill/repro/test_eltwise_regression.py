#!/usr/bin/env python3
"""CPU regression matrix for native TFDL elementwise operators.

The suite covers equal-shape and broadcast tensor/tensor execution, UINT8
qinfo semantics, saturation, non-eight-aligned tails, and FP16/FP32 paths.
It deliberately uses no model weights and no custom operators.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from TFDL2 import Op, TFContext, TFExecutor
from TFDL2.Common import TFDataType


NP_OPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Add": np.add,
    "Sub": np.subtract,
    "Mul": np.multiply,
    "Div": np.divide,
}


@dataclass
class Result:
    name: str
    status: str
    dtype: str
    operation: str
    shape_a: list[int]
    shape_b: list[int]
    max_abs: float | None = None
    max_code_diff: int | None = None
    detail: str = ""


def _context_name(name: str) -> str:
    return "EltwiseRegression_" + "".join(
        character if character.isalnum() else "_" for character in name
    )


def _scalar_qinfo(tensor: object) -> tuple[float, int]:
    scales = list(tensor.qscale)
    zero_points = list(tensor.qzeropoint)
    if len(scales) != 1 or len(zero_points) != 1:
        raise RuntimeError(
            f"expected scalar qinfo, got scales={scales}, zeros={zero_points}"
        )
    return float(scales[0]), int(zero_points[0])


def _dequantize(codes: np.ndarray, tensor: object) -> np.ndarray:
    scale, zero = _scalar_qinfo(tensor)
    return (np.asarray(codes, dtype=np.float64) - zero) * scale


def _quantize(real: np.ndarray, tensor: object) -> np.ndarray:
    scale, zero = _scalar_qinfo(tensor)
    # The final code coordinate is non-negative. floor(x + 0.5) matches the
    # nearest rounding used by the fixed-point path except at allowed 1-code
    # tie differences.
    codes = np.floor(np.asarray(real, dtype=np.float64) / scale + zero + 0.5)
    return np.clip(codes, 0, 255).astype(np.uint8)


def run_uint8(
    *,
    name: str,
    operation: str,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    values_a: np.ndarray,
    values_b: np.ndarray,
    range_a: tuple[float, float],
    range_b: tuple[float, float],
    range_y: tuple[float, float],
    frugal: bool = False,
    exact: bool = False,
) -> Result:
    context = TFContext(_context_name(name + f"_f{int(frugal)}"))
    try:
        with context:
            a = Op.Placeholder2(context, shape_a, TFDataType.TFDL_UINT8)
            b = Op.Placeholder2(context, shape_b, TFDataType.TFDL_UINT8)
            y = getattr(Op, operation)(a, b)
            for symbol, limits in ((a, range_a), (b, range_b), (y, range_y)):
                low, high = limits
                if not context.AddInt8Config(str(symbol), high, low):
                    raise RuntimeError(f"failed to register qinfo for {symbol}")
            context.SetOutputs([str(y)])
        executor = TFExecutor(
            context, {"UseHardware": False, "FrugalMode": frugal}
        )
        inputs = executor.GetInputs()
        for tensor, value in zip(inputs, (values_a, values_b), strict=True):
            tensor.fromNumpy(np.ascontiguousarray(value, dtype=np.uint8))
        output = executor()[0]
        actual = output.toNumpy().astype(np.uint8)
        real_a = _dequantize(values_a, inputs[0])
        real_b = _dequantize(values_b, inputs[1])
        expected = _quantize(NP_OPS[operation](real_a, real_b), output)
        difference = np.abs(
            actual.astype(np.int16) - expected.astype(np.int16)
        )
        maximum = int(difference.max(initial=0))
        tolerance = 0 if exact else 1
        return Result(
            name=name + f"/frugal={frugal}",
            status="PASS" if maximum <= tolerance else "FAIL",
            dtype="UINT8",
            operation=operation,
            shape_a=list(shape_a),
            shape_b=list(shape_b),
            max_code_diff=maximum,
            detail=(
                f"tolerance={tolerance}; mismatches="
                f"{int(np.count_nonzero(difference > tolerance))}"
            ),
        )
    except Exception as error:  # Keep the matrix running after one SDK error.
        return Result(
            name=name + f"/frugal={frugal}",
            status="ERROR",
            dtype="UINT8",
            operation=operation,
            shape_a=list(shape_a),
            shape_b=list(shape_b),
            detail=f"{type(error).__name__}: {error}",
        )


def run_float(
    *,
    name: str,
    operation: str,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    dtype: TFDataType,
) -> Result:
    context = TFContext(_context_name(name))
    dtype_name = "FP16" if dtype == TFDataType.TFDL_FLOAT16 else "FP32"
    numpy_dtype = np.float16 if dtype == TFDataType.TFDL_FLOAT16 else np.float32
    try:
        with context:
            a = Op.Placeholder2(context, shape_a, dtype)
            b = Op.Placeholder2(context, shape_b, dtype)
            y = getattr(Op, operation)(a, b)
            context.SetOutputs([str(y)])
        executor = TFExecutor(
            context, {"UseHardware": False, "FrugalMode": False}
        )
        values_a = np.linspace(
            -1.35, 1.65, int(np.prod(shape_a)), dtype=np.float32
        ).reshape(shape_a).astype(numpy_dtype)
        # Keep Div denominators away from zero.
        values_b = np.linspace(
            0.35, 1.75, int(np.prod(shape_b)), dtype=np.float32
        ).reshape(shape_b).astype(numpy_dtype)
        for tensor, value in zip(
            executor.GetInputs(), (values_a, values_b), strict=True
        ):
            tensor.fromNumpy(np.ascontiguousarray(value))
        actual = executor()[0].toNumpy().astype(np.float32)
        expected = NP_OPS[operation](values_a, values_b).astype(
            numpy_dtype
        ).astype(np.float32)
        maximum = float(np.max(np.abs(actual - expected), initial=0.0))
        tolerance = 2e-3 if dtype == TFDataType.TFDL_FLOAT16 else 2e-6
        return Result(
            name=name,
            status="PASS" if maximum <= tolerance else "FAIL",
            dtype=dtype_name,
            operation=operation,
            shape_a=list(shape_a),
            shape_b=list(shape_b),
            max_abs=maximum,
            detail=f"tolerance={tolerance:g}",
        )
    except Exception as error:
        return Result(
            name=name,
            status="ERROR",
            dtype=dtype_name,
            operation=operation,
            shape_a=list(shape_a),
            shape_b=list(shape_b),
            detail=f"{type(error).__name__}: {error}",
        )


def build_suite() -> list[Result]:
    results: list[Result] = []

    # Exact identity checks exercise both the vector body and scalar tail.
    for width in (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129):
        shape = (1, 1, 1, width)
        left = np.arange(width, dtype=np.uint8).reshape(shape)
        right = np.full(shape, 2, dtype=np.uint8)
        for frugal in (False, True):
            results.append(
                run_uint8(
                    name=f"u8_mul_identity_w{width}",
                    operation="Mul",
                    shape_a=shape,
                    shape_b=shape,
                    values_a=left,
                    values_b=right,
                    range_a=(0.0, 255.0),
                    range_b=(0.0, 127.5),
                    range_y=(0.0, 255.0),
                    frugal=frugal,
                    exact=True,
                )
            )

    rng = np.random.default_rng(20260814)
    broadcast_shapes = {
        "equal": ((1, 4, 3, 17), (1, 4, 3, 17)),
        "channel": ((1, 4, 3, 17), (1, 4, 1, 1)),
        "spatial": ((1, 4, 3, 17), (1, 1, 3, 17)),
        "tensor_scalar": ((1, 4, 3, 17), (1, 1, 1, 1)),
        "reverse_channel": ((1, 4, 1, 1), (1, 4, 3, 17)),
    }
    output_ranges = {
        "Add": (-4.0, 5.0),
        "Sub": (-5.0, 4.0),
        "Mul": (-5.0, 5.0),
        "Div": (-10.0, 10.0),
    }
    for operation in NP_OPS:
        for layout, (shape_a, shape_b) in broadcast_shapes.items():
            values_a = rng.integers(0, 256, size=shape_a, dtype=np.uint8)
            # Avoid a quantized zero in Div cases even if a positive-only
            # requested range is expanded by the SDK to include real zero.
            values_b = rng.integers(16, 256, size=shape_b, dtype=np.uint8)
            results.append(
                run_uint8(
                    name=f"u8_{operation.lower()}_{layout}",
                    operation=operation,
                    shape_a=shape_a,
                    shape_b=shape_b,
                    values_a=values_a,
                    values_b=values_b,
                    range_a=(-2.0, 2.0),
                    # A positive-only divisor also exercises a negative
                    # zero-point without introducing division by zero.
                    range_b=(0.25, 2.25),
                    range_y=output_ranges[operation],
                )
            )

    # Explicit signed zero-points plus output saturation.
    shape = (1, 1, 1, 17)
    results.append(
        run_uint8(
            name="u8_mul_signed_zp_saturation",
            operation="Mul",
            shape_a=shape,
            shape_b=shape,
            values_a=np.linspace(0, 255, 17, dtype=np.uint8).reshape(shape),
            values_b=np.full(shape, 132, dtype=np.uint8),
            range_a=(-128.0, 127.0),
            range_b=(-64.0, 63.5),
            # outputScale=1, so inputScaleA*inputScaleB/outputScale=0.5;
            # the end points still exercise UINT8 saturation after x2.
            range_y=(-128.0, 127.0),
        )
    )

    float_shapes = {
        "equal": ((1, 4, 3, 17), (1, 4, 3, 17)),
        "channel": ((1, 4, 3, 17), (1, 4, 1, 1)),
        "spatial": ((1, 4, 3, 17), (1, 1, 3, 17)),
        "tensor_scalar": ((1, 4, 3, 17), (1, 1, 1, 1)),
        "reverse_channel": ((1, 4, 1, 1), (1, 4, 3, 17)),
    }
    for dtype in (TFDataType.TFDL_FLOAT, TFDataType.TFDL_FLOAT16):
        dtype_name = "fp32" if dtype == TFDataType.TFDL_FLOAT else "fp16"
        for operation in NP_OPS:
            for layout, (shape_a, shape_b) in float_shapes.items():
                results.append(
                    run_float(
                        name=f"{dtype_name}_{operation.lower()}_{layout}",
                        operation=operation,
                        shape_a=shape_a,
                        shape_b=shape_b,
                        dtype=dtype,
                    )
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero if any case fails or raises an SDK error",
    )
    args = parser.parse_args()

    results = build_suite()
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("PASS", "FAIL", "ERROR")
    }
    report = {"summary": counts, "cases": [asdict(result) for result in results]}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        from pathlib import Path

        Path(args.output_json).write_text(rendered + "\n")
    if args.require_all and (counts["FAIL"] or counts["ERROR"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
