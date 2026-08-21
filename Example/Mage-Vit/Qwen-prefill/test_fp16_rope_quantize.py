#!/usr/bin/env python3
"""Regression test for the FP16 ApplyRope -> UINT8 Quantize boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/fp16_rope_quantize.fb")
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    args = parser.parse_args()
    Op, TFContext, TFExecutor, TFDataType = prefill._load_tfdl(
        args.addon_path
    )
    q_shape = (1, 32, 4, 128)
    k_shape = (1, 8, 4, 128)
    context = TFContext("Fp16RopeQuantizeRegression")
    with context:
        q = Op.Placeholder2(context, q_shape, TFDataType.TFDL_FLOAT16)
        k = Op.Placeholder2(context, k_shape, TFDataType.TFDL_FLOAT16)
        sin = Op.Placeholder2(
            context, (1, 1, 4, 128), TFDataType.TFDL_FLOAT
        )
        cos = Op.Placeholder2(
            context, (1, 1, 4, 128), TFDataType.TFDL_FLOAT
        )
        q_rope, k_rope = Op.Custom(
            (q, k, sin, cos), ("rope_q", "rope_k"), "ApplyRope", "{}"
        )
        q_u8 = Op.Quantize(q_rope)
        k_u8 = Op.Quantize(k_rope)
        output = Op.Concat((q_u8, k_u8), axis=1)
    for value in (q_u8, k_u8, output):
        if not context.AddInt8Config(str(value), 1.0, -1.0):
            raise RuntimeError(f"failed to register qinfo for {value}")
    context.SetOutputs([str(output)])
    target = args.output[:-3] if args.output.endswith(".fb") else args.output
    context.Dump(target)

    loaded_context = TFContext(path=args.output)
    executor = TFExecutor(
        loaded_context,
        {"UseHardware": False, "FrugalMode": False},
    )
    rng = np.random.default_rng(20260813)
    q_value = rng.normal(scale=0.2, size=q_shape).astype(np.float16)
    k_value = rng.normal(scale=0.2, size=k_shape).astype(np.float16)
    sin_value, cos_value = prefill.compute_rope(
        np.arange(4), 128, 1_000_000.0
    )
    for tensor, value in zip(
        executor.GetInputs(), (q_value, k_value, sin_value, cos_value)
    ):
        tensor.fromNumpy(np.ascontiguousarray(value))
    actual = executor()[0].toNumpy()
    print(
        f"FP16 ApplyRope -> Quantize passed: shape={actual.shape} "
        f"codes=[{actual.min()},{actual.max()}]"
    )


if __name__ == "__main__":
    main()
