#!/usr/bin/env python3
"""Execute a Qwen prefill FB and print every exposed tensor boundary.

This helper is intentionally reference-free: ``build_qwen_prefill.py
--debug-output`` can expose an intermediate tensor that no longer has the
normal hidden/K/V ABI, and this program still feeds the model correctly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fb", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--hidden-npy")
    parser.add_argument("--position-ids-npy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--print-info", action="store_true")
    parser.add_argument("--dump-prefix")
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return parser.parse_args()


def _feed(tensor: object, value: np.ndarray, data_type: object) -> None:
    from TFDL2.Common import TFDataType

    array = np.asarray(value)
    if data_type == TFDataType.TFDL_FLOAT16:
        array = array.astype(np.float16)
    elif data_type == TFDataType.TFDL_FLOAT:
        array = array.astype(np.float32)
    elif data_type == TFDataType.TFDL_UINT8:
        qmin = float(tensor.qmin[0])
        qmax = float(tensor.qmax[0])
        if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
            raise ValueError(
                f"invalid input qinfo for {tensor.name}: {qmin}, {qmax}"
            )
        array = np.clip(
            np.rint((array - qmin) * (255.0 / (qmax - qmin))), 0, 255
        ).astype(np.uint8)
    else:
        raise TypeError(f"unsupported input dtype {data_type}")
    tensor.fromNumpy(np.ascontiguousarray(array))


def main() -> None:
    args = parse_args()
    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    if args.hidden_npy:
        hidden = np.load(args.hidden_npy).astype(np.float32)
    else:
        hidden = np.random.default_rng(args.seed).normal(
            scale=args.input_scale,
            size=(1, args.seq_len, config.hidden_size),
        ).astype(np.float32)
    if args.position_ids_npy:
        positions = np.load(args.position_ids_npy).reshape(-1)
    else:
        positions = np.arange(args.seq_len, dtype=np.int64)
    sin, cos = prefill.compute_rope(
        positions, config.head_dim, config.rope_theta
    )
    mask = prefill.causal_mask(args.seq_len)

    _, TFContext, TFExecutor, _ = prefill._load_tfdl(args.addon_path)
    # Keep the shared-weight context alive for the executor lifetime.
    context = TFContext(path=args.fb)
    executor = TFExecutor(
        context,
        prefill.prefill_executor_config(
            bool(args.hardware),
            frugal_mode=False,
            software_attn_softmax_impl=False,
        ),
    )
    if args.print_info:
        executor.SetPrintInfo(True)
    inputs = executor.GetInputs()
    values = (hidden, sin, cos) if len(inputs) == 3 else (hidden, sin, cos, mask)
    if len(inputs) not in (3, 4):
        raise RuntimeError(f"expected 3 or 4 inputs, got {len(inputs)}")
    for tensor, value in zip(inputs, values):
        _feed(tensor, value, tensor.dtype)

    outputs = executor()
    report: list[dict[str, object]] = []
    for index, tensor in enumerate(outputs):
        array = tensor.toNumpy()
        item: dict[str, object] = {
            "index": index,
            "name": str(tensor.name),
            "dtype": str(tensor.dtype),
            "shape": list(array.shape),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array, dtype=np.float64)),
        }
        if hasattr(tensor, "qmin"):
            item["qmin_count"] = len(tensor.qmin)
            item["qmax_count"] = len(tensor.qmax)
            item["qmin_head"] = [float(x) for x in tensor.qmin[:4]]
            item["qmax_head"] = [float(x) for x in tensor.qmax[:4]]
        report.append(item)
        if args.dump_prefix:
            np.save(f"{args.dump_prefix}.{index}.npy", array)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
