#!/usr/bin/env python3
"""Collect ranges and compare a Qwen prefill layer against a TFDL artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(actual, dtype=np.float64).reshape(-1)
    error = got - ref
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(got))
    return {
        "cosine": float(np.dot(ref, got) / denominator) if denominator else 0.0,
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "rel_l2": float(np.linalg.norm(error) / np.linalg.norm(ref)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--fb", help="Optional float/quant TFDL layer artifact")
    parser.add_argument("--hidden-npy")
    parser.add_argument("--position-ids-npy")
    parser.add_argument("--dump-hidden-npy")
    parser.add_argument("--dump-ranges")
    parser.add_argument("--output-json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    weights = prefill.load_layer_weights(args.model_path, args.layer_id)
    if args.hidden_npy:
        hidden_np = np.load(args.hidden_npy).astype(np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        hidden_np = rng.normal(
            scale=args.input_scale,
            size=(1, args.seq_len, config.hidden_size),
        ).astype(np.float32)
    expected_hidden = (1, args.seq_len, config.hidden_size)
    if hidden_np.shape != expected_hidden:
        raise ValueError(
            f"hidden shape is {hidden_np.shape}, expected {expected_hidden}"
        )
    if args.position_ids_npy:
        position_ids = np.load(args.position_ids_npy).reshape(-1)
    else:
        position_ids = np.arange(args.seq_len, dtype=np.int64)
    if position_ids.shape != (args.seq_len,):
        raise ValueError(
            f"position_ids shape is {position_ids.shape}, expected {(args.seq_len,)}"
        )
    sin_np, cos_np = prefill.compute_rope(
        position_ids, config.head_dim, config.rope_theta
    )
    mask_np = prefill.causal_mask(args.seq_len)

    device = torch.device(args.device)
    hidden = torch.from_numpy(hidden_np).to(device)
    sin = torch.from_numpy(sin_np).to(device)
    cos = torch.from_numpy(cos_np).to(device)
    collector = prefill.RangeCollector()
    started = time.perf_counter()
    with torch.no_grad():
        reference = prefill.torch_layer(
            config,
            args.layer_id,
            weights,
            hidden,
            sin,
            cos,
            collector,
        )
    torch_seconds = time.perf_counter() - started
    reference_np = tuple(
        np.ascontiguousarray(value.detach().float().cpu().numpy())
        for value in reference
    )
    if args.dump_hidden_npy:
        np.save(args.dump_hidden_npy, reference_np[0])
    if args.dump_ranges:
        collector.dump(args.dump_ranges)

    report: dict[str, object] = {
        "model_path": args.model_path,
        "layer": args.layer_id,
        "seq_len": args.seq_len,
        "torch_device": str(device),
        "torch_seconds": torch_seconds,
        "range_count": len(collector.items),
        "reference_shapes": [list(value.shape) for value in reference_np],
    }
    if args.fb:
        _, TFContext, TFExecutor, _ = prefill._load_tfdl(args.addon_path)
        compile_started = time.perf_counter()
        # shareWeight=true makes TFContext the owner of executor weight data.
        context = TFContext(path=str(args.fb))
        executor = TFExecutor(
            context,
            prefill.prefill_executor_config(
                bool(args.hardware),
                software_attn_softmax_impl=True,
            ),
        )
        compile_seconds = time.perf_counter() - compile_started
        execute_started = time.perf_counter()
        actual = prefill.execute_layer(
            executor, hidden_np, sin_np, cos_np, mask_np
        )
        execute_seconds = time.perf_counter() - execute_started
        labels = ("hidden", "key", "value")
        comparisons = {
            label: {
                "reference_dtype": str(reference_value.dtype),
                "actual_dtype": str(actual_value.dtype),
                "shape": list(actual_value.shape),
                **metrics(reference_value, actual_value),
            }
            for label, reference_value, actual_value in zip(
                labels, reference_np, actual
            )
        }
        report.update(
            {
                "fb": args.fb,
                "use_hardware": bool(args.hardware),
                "compile_seconds": compile_seconds,
                "execute_seconds": execute_seconds,
                "comparisons": comparisons,
            }
        )
        for label in labels:
            item = comparisons[label]
            print(
                f"{label}: cosine={item['cosine']:.9f} "
                f"max_abs={item['max_abs']:.6g} rel_l2={item['rel_l2']:.6g}"
            )
        print(
            f"torch={torch_seconds:.3f}s compile={compile_seconds:.3f}s "
            f"execute={execute_seconds:.3f}s"
        )
    else:
        print(
            f"torch layer {args.layer_id}: {torch_seconds:.3f}s, "
            f"ranges={len(collector.items)}"
        )

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
