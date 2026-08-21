#!/usr/bin/env python3
"""Audit Qwen Q/K/V, H*S QK/Requant, Softmax and AV boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import qwen_prefill as prefill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--fb", required=True)
    parser.add_argument("--symbol-map", required=True)
    parser.add_argument("--prompt-dir")
    parser.add_argument("--input-hidden-npy")
    parser.add_argument("--position-ids-npy")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return parser.parse_args()


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(actual, dtype=np.float64).reshape(-1)
    error = right - left
    norm = float(np.linalg.norm(left))
    denominator = norm * float(np.linalg.norm(right))
    return {
        "cosine": float(np.dot(left, right) / denominator)
        if denominator else 0.0,
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "rel_l2": float(np.linalg.norm(error) / norm) if norm else 0.0,
    }


def _decode(tensor: Any, array: np.ndarray) -> np.ndarray:
    if array.dtype != np.uint8:
        return array.astype(np.float32)
    rows = array.reshape(-1, array.shape[-1]).astype(np.float32)
    scale = np.asarray(tensor.qscale, dtype=np.float32).reshape(-1)
    zero = np.asarray(tensor.qzeropoint, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        decoded = (rows - zero[0]) * scale[0]
    elif scale.size == rows.shape[0]:
        decoded = (rows - zero[:, None]) * scale[:, None]
    else:
        raise ValueError(
            f"{tensor.name}: qinfo={scale.size}, rows={rows.shape[0]}"
        )
    return decoded.reshape(array.shape)


class Capture:
    def __init__(self) -> None:
        self.values: dict[str, np.ndarray] = {}

    def observe(self, name: str, value: Any) -> Any:
        self.values[name] = np.ascontiguousarray(
            value.detach().float().cpu().numpy()
        )
        return value


def main() -> None:
    args = parse_args()
    import torch

    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    if args.input_hidden_npy:
        if not args.position_ids_npy:
            raise ValueError("--input-hidden-npy requires --position-ids-npy")
        hidden = np.load(args.input_hidden_npy).astype(np.float32)
        positions = np.load(args.position_ids_npy).reshape(-1)
    elif args.prompt_dir:
        root = Path(args.prompt_dir)
        hidden = np.load(root / "hidden.npy").astype(np.float32)
        positions = np.load(root / "position_ids.npy").reshape(-1)
    else:
        if args.seq_len is None or args.seq_len <= 0:
            raise ValueError("pass --prompt-dir or a positive --seq-len")
        hidden = np.random.default_rng(args.seed).normal(
            scale=0.1,
            size=(1, args.seq_len, config.hidden_size),
        ).astype(np.float32)
        positions = np.arange(args.seq_len, dtype=np.int64)
    sequence = positions.size
    sin, cos = prefill.compute_rope(
        positions, config.head_dim, config.rope_theta
    )
    weights = prefill.load_layer_weights(args.model_path, args.layer_id)
    device = torch.device(args.device)
    capture = Capture()
    with torch.inference_mode():
        reference_hidden, reference_key, reference_value = prefill.torch_layer(
            config,
            args.layer_id,
            weights,
            torch.from_numpy(hidden).to(device),
            torch.from_numpy(sin).to(device),
            torch.from_numpy(cos).to(device),
            capture,  # type: ignore[arg-type]
        )

    symbols = json.loads(Path(args.symbol_map).read_text())
    prefix = f"layers.{args.layer_id}"
    _, TFContext, TFExecutor, _ = prefill._load_tfdl(args.addon_path)
    context = TFContext(path=args.fb)
    executor = TFExecutor(
        context,
        {
            "UseHardware": False,
            "FrugalMode": False,
            "optimize": {"AttnSoftmaxImpl": False},
        },
    )
    actual_outputs = prefill.execute_layer(executor, hidden, sin, cos)

    def tensor(suffix: str) -> tuple[Any, np.ndarray, np.ndarray]:
        logical = f"{prefix}.{suffix}"
        if logical not in symbols:
            raise KeyError(f"symbol map is missing {logical}")
        value = executor.GetTensorByName(symbols[logical])
        raw = value.toNumpy()
        return value, raw, _decode(value, raw)

    comparisons: dict[str, dict[str, object]] = {}
    pairs = {
        "q_projection": ("self_attn.q_proj.fp16", "self_attn.q_proj"),
        "k_projection": ("self_attn.k_proj.fp16", "self_attn.k_proj"),
        "v_projection": ("self_attn.v_proj.fp16", "self_attn.v_proj"),
        "q_rope": ("self_attn.q_rope", "self_attn.q_rope"),
        "k_rope": ("self_attn.k_rope", "self_attn.k_rope"),
        "v_cache": ("self_attn.v_cache", "self_attn.v_cache"),
        "qk": ("self_attn.qk_matmul", "self_attn.qk_matmul"),
        "scores": ("self_attn.scores", "self_attn.scores"),
        "probability": (
            "self_attn.probabilities", "self_attn.probabilities"
        ),
        "av": ("self_attn.attention", "self_attn.attention"),
    }
    tensor_cache: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
    for label, (actual_suffix, reference_suffix) in pairs.items():
        actual_tensor, raw, decoded = tensor(actual_suffix)
        tensor_cache[actual_suffix] = (actual_tensor, raw, decoded)
        reference = capture.values[f"{prefix}.{reference_suffix}"]
        decoded = decoded.reshape(reference.shape)
        comparisons[label] = {
            **_metrics(reference, decoded),
            "dtype": str(actual_tensor.dtype),
            "shape": list(raw.shape),
            "qinfo_count": len(actual_tensor.qmin),
            "raw_min": int(raw.min()) if raw.dtype == np.uint8 else None,
            "raw_max": int(raw.max()) if raw.dtype == np.uint8 else None,
        }

    qk_tensor, qk_raw, _ = tensor_cache["self_attn.qk_matmul"]
    score_tensor, score_raw, _ = tensor_cache["self_attn.scores"]
    transport_logical = f"{prefix}.self_attn.scores.transport"
    probability_tensor, probability_raw, _ = tensor_cache[
        "self_attn.probabilities"
    ]
    attention_tensor, _, _ = tensor_cache["self_attn.attention"]
    qk_scale = np.asarray(qk_tensor.qscale, dtype=np.float64)
    score_scale = np.asarray(score_tensor.qscale, dtype=np.float64)
    expected_score_scale = qk_scale / np.sqrt(config.head_dim)
    upper = np.broadcast_to(
        np.triu(np.ones((sequence, sequence), dtype=bool), 1),
        (config.num_attention_heads, sequence, sequence),
    )
    probability_codes = probability_raw.reshape(
        config.num_attention_heads, sequence, sequence
    )
    probability_zero = int(probability_tensor.qzeropoint[0])
    graph_checks = {
        "input_count": len(executor.GetInputs()),
        "has_causal_mask_input": len(executor.GetInputs()) == 4,
        "qk_qinfo_count": len(qk_tensor.qmin),
        "score_requant_qinfo_count": len(score_tensor.qmin),
        "custom_input_qinfo_count": len(score_tensor.qmin),
        "has_score_transport_symbol": transport_logical in symbols,
        "qk_to_requant_raw_bit_exact": bool(
            np.array_equal(qk_raw, score_raw)
        ),
        "score_scale_max_abs_error": float(
            np.max(np.abs(score_scale - expected_score_scale))
        ),
        "score_scale_max_relative_error": float(
            np.max(
                np.abs(score_scale - expected_score_scale)
                / expected_score_scale
            )
        ),
        "causal_upper_triangle_all_zero_point": bool(
            np.all(probability_codes[upper] == probability_zero)
        ),
        "probability_qinfo_count": len(probability_tensor.qmin),
        "probability_dtype": str(probability_tensor.dtype),
        "av_dtype": str(attention_tensor.dtype),
        "hidden_output_dtype": str(executor()[0].dtype),
        "k_output_dtype": str(executor()[1].dtype),
        "v_output_dtype": str(executor()[2].dtype),
        "post_quant_tensor_count": sum(
            name.startswith(("PostQuant_", "PostDeQuant_"))
            for name in context.GetAllTensorNames()
        ),
    }
    output_comparisons = {
        "hidden": _metrics(
            reference_hidden.detach().float().cpu().numpy(),
            actual_outputs[0],
        ),
        "key_export": _metrics(
            reference_key.detach().float().cpu().numpy(), actual_outputs[1]
        ),
        "value_export": _metrics(
            reference_value.detach().float().cpu().numpy(), actual_outputs[2]
        ),
    }
    report = {
        "model_path": args.model_path,
        "fb": args.fb,
        "layer": args.layer_id,
        "seq_len": sequence,
        "comparisons": comparisons,
        "outputs": output_comparisons,
        "graph_checks": graph_checks,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
