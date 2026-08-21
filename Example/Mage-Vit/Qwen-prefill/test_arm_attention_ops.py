#!/usr/bin/env python3
"""Correctness and timing checks for the ARM Qwen attention custom ops."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


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


def _encode_rows(
    values: np.ndarray, row_min: np.ndarray, row_max: np.ndarray
) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    low = row_min.astype(np.float64)[:, None]
    high = row_max.astype(np.float64)[:, None]
    scale = (high - low) / 255.0
    zero = np.clip(np.rint(-low / scale), 0.0, 255.0)
    return np.clip(np.rint(flat / scale + zero), 0.0, 255.0).astype(
        np.uint8
    ).reshape(values.shape)


def _decode_rows(
    codes: np.ndarray, row_min: np.ndarray, row_max: np.ndarray
) -> np.ndarray:
    flat = codes.reshape(-1, codes.shape[-1]).astype(np.float64)
    low = row_min.astype(np.float64)[:, None]
    high = row_max.astype(np.float64)[:, None]
    scale = (high - low) / 255.0
    zero = np.clip(np.rint(-low / scale), 0.0, 255.0)
    return ((flat - zero) * scale).reshape(codes.shape).astype(np.float32)


def _reference_codes(scores: np.ndarray) -> np.ndarray:
    sequence = scores.shape[-1]
    result = np.zeros(scores.shape, dtype=np.uint8)
    flat = scores.reshape(-1, sequence).astype(np.float64)
    output = result.reshape(-1, sequence)
    for row, values in enumerate(flat):
        query = row % sequence
        valid = values[: query + 1]
        exponent = np.exp(valid - np.max(valid))
        probability = exponent / np.sum(exponent)
        output[row, : query + 1] = np.clip(
            np.rint(probability * 255.0), 0.0, 255.0
        ).astype(np.uint8)
    return result


def build_arm_softmax(
    tfdl, shape: tuple[int, ...], row_min: np.ndarray,
    row_max: np.ndarray, threads: int, per_row: bool,
):
    Op, TFContext, TFExecutor, TFDataType = tfdl
    context = TFContext(
        f"ArmCausalMaskSoftmaxTestS{shape[-1]}T{threads}R{int(per_row)}"
    )
    with context:
        source = Op.Placeholder2(context, shape, TFDataType.TFDL_UINT8)
        output = _custom_output(
            Op.Custom(
                (source,),
                (f"arm_causal_softmax_s{shape[-1]}_t{threads}",),
                "ArmCausalMaskSoftmax",
                json.dumps({"threads": int(threads)}),
            )
        )
    if per_row:
        # Directly exercise the SDK's CustomOp H*S-qinfo scheduling path.
        assert context.AddInt8ConfigPerChannel(
            str(source), row_max.tolist(), row_min.tolist()
        )
    else:
        assert context.AddInt8Config(
            str(source), float(row_max[0]), float(row_min[0])
        )
    assert context.AddInt8Config(str(output), 1.0, 0.0)
    context.SetOutputs([str(output)])
    executor = TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )
    return executor


def build_fp16_masksoftmax(tfdl, shape: tuple[int, ...]):
    Op, TFContext, TFExecutor, TFDataType = tfdl
    context = TFContext(f"Fp16AddSoftmaxQuantizeBaselineS{shape[-1]}")
    sequence = shape[-1]
    mask = np.zeros(shape, dtype=np.float16)
    mask[..., np.triu_indices(sequence, 1)[0],
         np.triu_indices(sequence, 1)[1]] = np.float16(-65504.0)
    context.RegisterParamToContext(causal_mask=mask)
    with context:
        source = Op.Placeholder2(context, shape, TFDataType.TFDL_FLOAT16)
        masked = Op.Add(source, context.GetParamSymbol("causal_mask"))
        probability = Op.Softmax(masked, axis=len(shape) - 1)
        output = Op.Quantize(probability)
    if not context.AddInt8Config(str(output), 1.0, 0.0):
        raise RuntimeError(
            "failed to register FP16 Add/Softmax output quantization"
        )
    context.SetOutputs([str(output)])
    return TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )


def softmax_case(
    tfdl, heads: int, sequence: int, threads: int,
    per_row: bool, iterations: int, seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + sequence + threads + int(per_row))
    shape = (heads, sequence, sequence)
    rows = heads * sequence
    if per_row:
        scales = rng.uniform(0.005, 0.08, size=rows).astype(np.float32)
        zeros = rng.integers(32, 224, size=rows)
    else:
        scales = np.full(rows, 0.035, dtype=np.float32)
        zeros = np.full(rows, 127, dtype=np.int64)
    row_min = -scales * zeros
    row_max = scales * (255 - zeros)
    codes = rng.integers(0, 256, size=shape, dtype=np.uint8)
    scores = _decode_rows(codes, row_min, row_max)
    reference = _reference_codes(scores)

    executor = build_arm_softmax(
        tfdl, shape, row_min, row_max, threads, per_row
    )
    input_tensor = executor.GetInputs()[0]
    input_tensor.fromNumpy(np.ascontiguousarray(codes))
    actual = executor()[0].toNumpy()
    difference = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    upper_mask = np.broadcast_to(
        np.triu(np.ones((sequence, sequence), dtype=bool), k=1), shape
    )
    if np.any(actual[upper_mask] != 0):
        raise AssertionError("causal upper triangle is not output zero-point")
    if int(np.max(difference)) > 1:
        location = np.unravel_index(int(np.argmax(difference)), shape)
        raise AssertionError(
            f"S={sequence} per_row={per_row}: max code difference "
            f"{int(np.max(difference))} at {location}"
        )

    for _ in range(2):
        executor()
    started = time.perf_counter()
    for _ in range(iterations):
        executor()
    arm_seconds = (time.perf_counter() - started) / iterations

    baseline = build_fp16_masksoftmax(tfdl, shape)
    baseline.GetInputs()[0].fromNumpy(np.ascontiguousarray(scores, dtype=np.float16))
    for _ in range(2):
        baseline()
    started = time.perf_counter()
    for _ in range(iterations):
        baseline()
    fp16_seconds = (time.perf_counter() - started) / iterations
    return {
        "heads": heads,
        "sequence": sequence,
        "threads": threads,
        "per_row_qinfo": per_row,
        "elements": int(np.prod(shape)),
        "max_code_difference": int(np.max(difference)),
        "different_code_fraction": float(np.mean(difference != 0)),
        "arm_seconds": arm_seconds,
        "fp16_add_softmax_quantize_seconds": fp16_seconds,
        "speedup_vs_fp16_add_softmax_quantize": fp16_seconds / arm_seconds,
    }


def rope_case(tfdl, seed: int) -> dict[str, object]:
    Op, TFContext, TFExecutor, TFDataType = tfdl
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(1, 4, 7, 16)).astype(np.float16)
    k = rng.normal(size=(1, 2, 7, 16)).astype(np.float16)
    positions = np.arange(7, dtype=np.float32)
    inv = 1.0 / (10_000.0 ** (np.arange(0, 16, 2) / 16.0))
    angle = np.concatenate(
        (np.outer(positions, inv), np.outer(positions, inv)), axis=-1
    ).astype(np.float32)
    sin = np.sin(angle)[None, None].astype(np.float32)
    cos = np.cos(angle)[None, None].astype(np.float32)

    def reference(value: np.ndarray) -> np.ndarray:
        source = value.astype(np.float32)
        half = source.shape[-1] // 2
        rotated = np.concatenate(
            (-source[..., half:], source[..., :half]), axis=-1
        )
        return (source * cos + rotated * sin).astype(np.float16)

    context = TFContext("ApplyRopeNativeFp16Test")
    with context:
        q_input = Op.Placeholder2(context, q.shape, TFDataType.TFDL_FLOAT16)
        k_input = Op.Placeholder2(context, k.shape, TFDataType.TFDL_FLOAT16)
        sin_input = Op.Placeholder2(context, sin.shape, TFDataType.TFDL_FLOAT)
        cos_input = Op.Placeholder2(context, cos.shape, TFDataType.TFDL_FLOAT)
        outputs = Op.Custom(
            (q_input, k_input, sin_input, cos_input),
            ("rope_fp16_q", "rope_fp16_k"),
            "ApplyRope",
            "{}",
        )
    context.SetOutputs([str(outputs[0]), str(outputs[1])])
    executor = TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )
    for tensor, value in zip(executor.GetInputs(), (q, k, sin, cos)):
        tensor.fromNumpy(np.ascontiguousarray(value))
    actual_q, actual_k = (value.toNumpy() for value in executor())
    reference_q = reference(q)
    reference_k = reference(k)
    return {
        "q_max_abs": float(
            np.max(np.abs(actual_q.astype(np.float32) - reference_q.astype(np.float32)))
        ),
        "k_max_abs": float(
            np.max(np.abs(actual_k.astype(np.float32) - reference_k.astype(np.float32)))
        ),
        "q_exact_fraction": float(np.mean(actual_q == reference_q)),
        "k_exact_fraction": float(np.mean(actual_k == reference_k)),
    }


def attention_case(tfdl, seed: int) -> dict[str, object]:
    """Exercise the production Custom Softmax -> UINT8 AV boundary."""
    Op, TFContext, TFExecutor, TFDataType = tfdl
    heads, sequence, dimension = 4, 4, 16
    rng = np.random.default_rng(seed + 91)
    scores = rng.integers(
        0, 256, size=(heads, sequence, sequence), dtype=np.uint8
    )
    value = rng.integers(
        0, 256, size=(heads, sequence, dimension), dtype=np.uint8
    )
    context = TFContext("ArmCausalMaskSoftmaxAvRegression")
    with context:
        score_input = Op.Placeholder2(
            context, scores.shape, TFDataType.TFDL_UINT8
        )
        value_input = Op.Placeholder2(
            context, value.shape, TFDataType.TFDL_UINT8
        )
        probability = _custom_output(
            Op.Custom(
                (score_input,),
                ("arm_causal_probability_for_av",),
                "ArmCausalMaskSoftmax",
                '{"threads":0}',
            )
        )
        output = Op.MatMul(
            probability, value_input, transA=False, transB=False
        )
    assert context.AddInt8Config(str(score_input), 4.0, -4.0)
    assert context.AddInt8Config(str(value_input), 1.0, -1.0)
    assert context.AddInt8Config(str(probability), 1.0, 0.0)
    assert context.AddInt8Config(str(output), 1.0, -1.0)
    context.SetOutputs([str(output)])
    executor = TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )
    for tensor, data in zip(executor.GetInputs(), (scores, value)):
        tensor.fromNumpy(np.ascontiguousarray(data))
    actual = executor()[0].toNumpy()
    return {
        "shape": list(actual.shape),
        "code_min": int(actual.min()),
        "code_max": int(actual.max()),
    }


def hxs_qk_attention_case(tfdl, seed: int) -> dict[str, object]:
    """Exercise QK(H*S) -> identity Requant -> custom Softmax -> AV."""
    Op, TFContext, TFExecutor, TFDataType = tfdl
    heads, sequence, dimension = 32, 4, 128
    rng = np.random.default_rng(seed + 92)
    q = rng.integers(
        0, 256, size=(heads, sequence, dimension), dtype=np.uint8
    )
    k = rng.integers(
        0, 256, size=(heads, dimension, sequence), dtype=np.uint8
    )
    value = rng.integers(
        0, 256, size=(heads, sequence, dimension), dtype=np.uint8
    )
    rows = heads * sequence
    qk_limit = np.linspace(64.0, 128.0, rows, dtype=np.float32)
    qk_min = -qk_limit
    qk_max = qk_limit
    context = TFContext("HxsQkArmAttentionRegression")
    with context:
        q_input = Op.Placeholder2(context, q.shape, TFDataType.TFDL_UINT8)
        k_input = Op.Placeholder2(context, k.shape, TFDataType.TFDL_UINT8)
        value_input = Op.Placeholder2(
            context, value.shape, TFDataType.TFDL_UINT8
        )
        qk = Op.MatMul(q_input, k_input, transA=False, transB=False)
        scores = Op.Requantize(qk, list(range(256)))
        probability = _custom_output(
            Op.Custom(
                (scores,),
                ("hxs_qk_arm_probability",),
                "ArmCausalMaskSoftmax",
                '{"threads":0}',
            )
        )
        output = Op.MatMul(
            probability, value_input, transA=False, transB=False
        )
    assert context.AddInt8Config(str(q_input), 1.0, -1.0)
    assert context.AddInt8Config(str(k_input), 1.0, -1.0)
    assert context.AddInt8Config(str(value_input), 1.0, -1.0)
    assert context.AddInt8ConfigPerChannel(
        str(qk), qk_max.tolist(), qk_min.tolist()
    )
    score_min = qk_min / np.sqrt(dimension)
    score_max = qk_max / np.sqrt(dimension)
    assert context.AddInt8ConfigPerChannel(
        str(scores), score_max.tolist(), score_min.tolist()
    )
    assert context.AddInt8Config(str(probability), 1.0, 0.0)
    assert context.AddInt8Config(str(output), 1.0, -1.0)
    context.SetOutputs([str(output)])
    executor = TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )
    for tensor, data in zip(executor.GetInputs(), (q, k, value)):
        tensor.fromNumpy(np.ascontiguousarray(data))
    actual = executor()[0].toNumpy()
    return {
        "shape": list(actual.shape),
        "code_min": int(actual.min()),
        "code_max": int(actual.max()),
        "qk_qinfo_count": rows,
    }


def rope_hxs_attention_case(tfdl, seed: int) -> dict[str, object]:
    """Exercise the complete FP16 RoPE entry through UINT8 AV."""
    Op, TFContext, TFExecutor, TFDataType = tfdl
    q_heads, kv_heads, sequence, dimension = 32, 8, 4, 128
    rng = np.random.default_rng(seed + 93)
    q = rng.normal(
        scale=0.2, size=(1, q_heads, sequence, dimension)
    ).astype(np.float16)
    k = rng.normal(
        scale=0.2, size=(1, kv_heads, sequence, dimension)
    ).astype(np.float16)
    value = rng.normal(
        scale=0.2, size=(1, kv_heads, sequence, dimension)
    ).astype(np.float16)
    sin, cos = prefill.compute_rope(
        np.arange(sequence), dimension, 1_000_000.0
    )
    rows = q_heads * sequence
    qk_min = np.full(rows, -16.0, dtype=np.float32)
    qk_max = np.full(rows, 16.0, dtype=np.float32)
    context = TFContext("RopeHxsArmAttentionRegression")
    with context:
        q_input = Op.Placeholder2(context, q.shape, TFDataType.TFDL_FLOAT16)
        k_input = Op.Placeholder2(context, k.shape, TFDataType.TFDL_FLOAT16)
        value_input = Op.Placeholder2(
            context, value.shape, TFDataType.TFDL_FLOAT16
        )
        sin_input = Op.Placeholder2(
            context, sin.shape, TFDataType.TFDL_FLOAT
        )
        cos_input = Op.Placeholder2(
            context, cos.shape, TFDataType.TFDL_FLOAT
        )
        q_rope, k_rope = Op.Custom(
            (q_input, k_input, sin_input, cos_input),
            ("rope_hxs_q", "rope_hxs_k"),
            "ApplyRope",
            "{}",
        )
        q_u8 = Op.Quantize(q_rope)
        k_u8 = Op.Quantize(k_rope)
        v_u8 = Op.Quantize(value_input)
        q3 = Op.Reshape(q_u8, (kv_heads, 4 * sequence, dimension))
        k3 = Op.Transpose(
            Op.Reshape(k_u8, (kv_heads, sequence, dimension)),
            (0, 2, 1),
        )
        v3 = Op.Reshape(v_u8, (kv_heads, sequence, dimension))
        qk_grouped = Op.MatMul(q3, k3, transA=False, transB=False)
        qk = qk_grouped
        scores = Op.Requantize(qk, list(range(256)))
        probability = _custom_output(
            Op.Custom(
                (scores,),
                ("rope_hxs_probability",),
                "ArmCausalMaskSoftmax",
                '{"threads":0}',
            )
        )
        probability_grouped = probability
        output_grouped = Op.MatMul(
            probability_grouped, v3, transA=False, transB=False
        )
        output = Op.Reshape(
            output_grouped, (q_heads, sequence, dimension)
        )
    for symbol in (q_u8, k_u8, v_u8, q3, k3, v3):
        assert context.AddInt8Config(str(symbol), 1.0, -1.0)
    for symbol in (qk_grouped, qk):
        assert context.AddInt8ConfigPerChannel(
            str(symbol), qk_max.tolist(), qk_min.tolist()
        )
    score_limit = float(16.0 / np.sqrt(dimension))
    score_min = np.full(rows, -score_limit, dtype=np.float32)
    score_max = np.full(rows, score_limit, dtype=np.float32)
    assert context.AddInt8ConfigPerChannel(
        str(scores), score_max.tolist(), score_min.tolist()
    )
    assert context.AddInt8Config(str(probability), 1.0, 0.0)
    assert context.AddInt8Config(str(probability_grouped), 1.0, 0.0)
    assert context.AddInt8Config(str(output_grouped), 1.0, -1.0)
    assert context.AddInt8Config(str(output), 1.0, -1.0)
    context.SetOutputs([str(output)])
    executor = TFExecutor(
        context, {"UseHardware": False, "FrugalMode": False}
    )
    for tensor, data in zip(
        executor.GetInputs(), (q, k, value, sin, cos)
    ):
        tensor.fromNumpy(np.ascontiguousarray(data))
    actual = executor()[0].toNumpy()
    return {
        "shape": list(actual.shape),
        "code_min": int(actual.min()),
        "code_max": int(actual.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, action="append")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--threads", type=int, action="append")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--addon-path",
        default=str(SDK_ROOT / "AddonOps/build/libTFDLAddOn.so"),
    )
    args = parser.parse_args()
    tfdl = _load_tfdl(Path(args.addon_path))
    sequences = args.sequence or [4, 128, 512]
    threads = args.threads or [0]
    report: dict[str, object] = {
        "rope_fp16": rope_case(tfdl, args.seed),
        "softmax_av": attention_case(tfdl, args.seed),
        "hxs_qk_softmax_av": hxs_qk_attention_case(tfdl, args.seed),
        "rope_hxs_softmax_av": rope_hxs_attention_case(tfdl, args.seed),
        "softmax": [],
    }
    for sequence in sequences:
        for thread_count in threads:
            for per_row in (False, True):
                item = softmax_case(
                    tfdl,
                    args.heads,
                    sequence,
                    thread_count,
                    per_row,
                    args.iterations,
                    args.seed,
                )
                report["softmax"].append(item)  # type: ignore[union-attr]
                print(json.dumps(item, sort_keys=True), flush=True)
    print(json.dumps(report["rope_fp16"], sort_keys=True), flush=True)
    print(json.dumps(report["softmax_av"], sort_keys=True), flush=True)
    print(
        json.dumps(report["hxs_qk_softmax_av"], sort_keys=True),
        flush=True,
    )
    print(
        json.dumps(report["rope_hxs_softmax_av"], sort_keys=True),
        flush=True,
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
