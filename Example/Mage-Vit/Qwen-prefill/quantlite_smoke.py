#!/usr/bin/env python3
"""Exercise QuantizeLite on an explicit Quantize/DeQuantize island."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--islands", type=int, default=1)
    parser.add_argument(
        "--dequant-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--input-dtype",
        choices=("fp16", "fp32"),
        default="fp32",
    )
    parser.add_argument("--rmsnorm-custom", action="store_true")
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    args = parser.parse_args()

    from TFDL2 import CalibrationMode, Op, TFCalibration, TFContext, TFExecutor
    from TFDL2.Common import TFDataType
    if args.rmsnorm_custom:
        from TFDL2.utils import LoadCustomOp

        LoadCustomOp(args.addon_path)
    dequant_dtype = (
        TFDataType.TFDL_FLOAT16
        if args.dequant_dtype == "fp16"
        else TFDataType.TFDL_FLOAT
    )
    input_dtype = (
        TFDataType.TFDL_FLOAT16
        if args.input_dtype == "fp16"
        else TFDataType.TFDL_FLOAT
    )

    rng = np.random.default_rng(20260811)
    input_value = rng.normal(0.0, 0.25, (1, 4, 1, 8)).astype(np.float32)
    weights = {
        f"conv.{index}.weight": rng.normal(
            0.0, 0.2, (4, 4, 1, 1)
        ).astype(np.float32)
        for index in range(args.islands)
    }
    reference_hidden = input_value.astype(np.float32)
    if args.rmsnorm_custom:
        reference_hidden = reference_hidden / np.sqrt(
            np.mean(reference_hidden**2, axis=-1, keepdims=True) + 1e-5
        )
    references: list[np.ndarray] = []
    for weight in weights.values():
        reference_hidden = np.einsum(
            "nchw,ocij->nohw", reference_hidden, weight
        )
        references.append(reference_hidden)
    reference = input_value.astype(np.float32) + reference_hidden

    context = TFContext("QuantizeLiteExplicitQDQSmoke")
    context.RegisterParamToContext(**weights)

    def add_range(symbol: object, array: np.ndarray) -> None:
        low = float(np.min(array))
        high = float(np.max(array))
        if not context.AddInt8Config(str(symbol), high, low):
            raise RuntimeError(f"AddInt8Config failed for {symbol}")

    with context:
        input_symbol = Op.Placeholder2(
            context, input_value.shape, input_dtype
        )
        add_range(input_symbol, input_value)
        hidden = input_symbol
        if args.rmsnorm_custom:
            hidden = Op.Custom(
                (hidden,), ("quantlite_smoke_rmsnorm",), "RMSNorm", "{}"
            )
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            add_range(hidden, reference_hidden)
        for index, reference_hidden in enumerate(references):
            quantized = Op.Quantize(hidden)
            add_range(quantized, input_value if index == 0 else references[index - 1])
            conv = Op.Convolution2(
                quantized,
                context.GetParamSymbol(f"conv.{index}.weight"),
                None,
                kernel=1,
                pad=0,
                stride=1,
                dilation=1,
                outChannel=4,
                group=1,
            )
            add_range(conv, reference_hidden)
            hidden = Op.DeQuantize(conv, dequant_dtype)
        residual = (
            Op.Cast(input_symbol, dequant_dtype)
            if dequant_dtype == TFDataType.TFDL_FLOAT16
            else input_symbol
        )
        output = Op.Add(residual, hidden)
    context.SetOutputs([str(output)])

    calibration = TFCalibration(
        context,
        CalibrationMode.Naive,
        {"UseHardware": False, "FrugalMode": True},
    )
    calibration.QuantizeLite(
        {str(input_symbol): input_dtype},
        stopquanttensors=(),
        MergeConcate=False,
        Perchannel=True,
    )

    print(
        "QuantizeLite weights:",
        [str(context.GetParam(name).dtype) for name in weights],
        flush=True,
    )
    output_path = args.output[:-3] if args.output.endswith(".fb") else args.output
    context.Dump(output_path)

    loaded_context = TFContext(path=str(args.output))
    executor = TFExecutor(
        loaded_context,
        {"UseHardware": False, "FrugalMode": True},
    )
    tensor = executor.GetInputs()[0]
    tensor.fromNumpy(np.ascontiguousarray(input_value))
    actual = executor()[0].toNumpy().astype(np.float32)
    reference_flat = reference.reshape(-1).astype(np.float64)
    actual_flat = actual.reshape(-1).astype(np.float64)
    metric = float(
        np.dot(reference_flat, actual_flat)
        / (np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat))
    )
    print(
        f"QuantizeLite explicit-QDQ smoke islands={args.islands} "
        f"input_dtype={args.input_dtype} "
        f"rmsnorm_custom={args.rmsnorm_custom} "
        f"dequant_dtype={args.dequant_dtype}: "
        f"cosine={metric:.9f} "
        f"max_abs={np.max(np.abs(reference - actual)):.6g} "
        f"input_dtype={tensor.dtype} output={args.output}"
    )


if __name__ == "__main__":
    main()
