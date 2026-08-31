#!/usr/bin/env python3
"""Export fixed-S GELab text prefill layers as pure FP16 TFDL graphs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
from project_compat import prepend_shared_paths  # noqa: E402

prepend_shared_paths()
import qwen_prefill as prefill  # noqa: E402


PROJECTION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_fp16_matmul_projections(
    artifact: Path,
    symbols: dict[str, str],
    layer: int,
    source_weights: dict[str, Any],
    config: Any,
) -> dict[str, object]:
    """Audit FP16 A@W projections and non-materialized grouped GQA."""
    from TFDL2 import TFContext

    context = TFContext(path=str(artifact.resolve()))
    prefix = f"layers.{layer}"
    rows: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    for suffix in PROJECTION_SUFFIXES:
        logical = f"{prefix}.{suffix}"
        weight_name = f"{logical}.weight"
        node_name = symbols[f"{logical}.matmul"]
        left_name = symbols[f"{logical}.matmul_input"]
        parameter_name = symbols[f"{logical}.matmul_weight"]
        attribute = json.loads(context._GetAttr(node_name))
        inputs = [str(value) for value in attribute.get("input", [])]
        parameters = attribute.get("param", {})
        tensor = context.GetParam(parameter_name)
        source_shape = list(source_weights[weight_name].shape)
        expected_shape = [source_shape[1], source_shape[0]]
        reasons: list[str] = []
        if attribute.get("layerType") != "MatMul":
            reasons.append(f"expected MatMul, got {attribute.get('layerType')}")
        if inputs != [left_name, parameter_name]:
            reasons.append(
                "expected [activation, weight] inputs "
                f"[{left_name}, {parameter_name}], got {inputs}"
            )
        if bool(parameters.get("transA")):
            reasons.append("transA must be false")
        if bool(parameters.get("transB")):
            reasons.append("transB must be false")
        if "FLOAT16" not in str(tensor.dtype):
            reasons.append(f"weight must be FLOAT16, got {tensor.dtype}")
        if list(tensor.shape) != expected_shape:
            reasons.append(
                f"weight must be pre-transposed [K,N]={expected_shape}, "
                f"got {list(tensor.shape)}"
            )
        row: dict[str, object] = {
            "projection": logical,
            "operator": attribute.get("layerType"),
            "node": node_name,
            "inputs": inputs,
            "activation_is_left_input": bool(inputs and inputs[0] == left_name),
            "weight_is_right_input": bool(
                len(inputs) == 2 and inputs[1] == parameter_name
            ),
            "trans_a": bool(parameters.get("transA")),
            "trans_b": bool(parameters.get("transB")),
            "weight": parameter_name,
            "weight_dtype": str(tensor.dtype),
            "weight_shape": list(tensor.shape),
            "expected_weight_shape": expected_shape,
            "valid": not reasons,
        }
        if reasons:
            row["reason"] = "; ".join(reasons)
            invalid.append(dict(row))
        rows.append(row)
    gqa_reasons: list[str] = []
    repeated_symbols = [
        logical
        for logical in (
            f"{prefix}.self_attn.k_repeated",
            f"{prefix}.self_attn.v_repeated",
        )
        if logical in symbols
    ]
    if repeated_symbols:
        gqa_reasons.append(
            "materialized K/V repeat symbols remain: " + ", ".join(repeated_symbols)
        )
    grouped_nodes = {
        "q_input": (f"{prefix}.self_attn.q_matmul_input", "Reshape"),
        "k_input": (f"{prefix}.self_attn.k_matmul_input", "Transpose"),
        "v_input": (f"{prefix}.self_attn.v_matmul_input", "Reshape"),
        "qk_grouped": (f"{prefix}.self_attn.qk_matmul.grouped", "MatMul"),
        "qk_heads": (f"{prefix}.self_attn.qk_matmul", "Reshape"),
        "probabilities_grouped": (
            f"{prefix}.self_attn.probabilities.grouped",
            "Reshape",
        ),
        "attention_grouped": (
            f"{prefix}.self_attn.attention.grouped",
            "MatMul",
        ),
        "attention_heads": (f"{prefix}.self_attn.attention", "Reshape"),
    }
    grouped_attributes: dict[str, dict[str, object]] = {}
    for role, (logical, expected_type) in grouped_nodes.items():
        node = symbols.get(logical)
        if node is None:
            gqa_reasons.append(f"missing grouped GQA symbol {logical}")
            continue
        attribute = json.loads(context._GetAttr(node))
        grouped_attributes[role] = attribute
        if attribute.get("layerType") != expected_type:
            gqa_reasons.append(
                f"{logical} must be {expected_type}, got {attribute.get('layerType')}"
            )
    qk_attribute = grouped_attributes.get("qk_grouped", {})
    qk_inputs = [str(value) for value in qk_attribute.get("input", [])]
    expected_qk_inputs = [
        symbols.get(f"{prefix}.self_attn.q_matmul_input"),
        symbols.get(f"{prefix}.self_attn.k_matmul_input"),
    ]
    if qk_inputs != expected_qk_inputs:
        gqa_reasons.append(
            f"grouped QK inputs must be {expected_qk_inputs}, got {qk_inputs}"
        )
    attention_attribute = grouped_attributes.get("attention_grouped", {})
    attention_inputs = [
        str(value) for value in attention_attribute.get("input", [])
    ]
    expected_attention_inputs = [
        symbols.get(f"{prefix}.self_attn.probabilities.grouped"),
        symbols.get(f"{prefix}.self_attn.v_matmul_input"),
    ]
    if attention_inputs != expected_attention_inputs:
        gqa_reasons.append(
            "grouped AV inputs must be "
            f"{expected_attention_inputs}, got {attention_inputs}"
        )
    gqa_audit = {
        "mode": "group-query-heads-in-m-dimension",
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "kv_repeats": config.kv_repeats,
        "materialized_kv_repeat": False,
        "materialized_repeat_symbols": repeated_symbols,
        "grouped_qk_operator": qk_attribute.get("layerType"),
        "grouped_av_operator": attention_attribute.get("layerType"),
        "valid": not gqa_reasons,
    }
    if gqa_reasons:
        gqa_audit["reason"] = "; ".join(gqa_reasons)
        invalid.append(
            {
                "projection": f"{prefix}.self_attn.grouped_gqa",
                "valid": False,
                "reason": gqa_audit["reason"],
            }
        )
    return {
        "format": "gelab-prefill-fp16-matmul-audit-v2",
        "fb": str(artifact.resolve()),
        "layer": layer,
        "main_trunk_layout": "BSC",
        "projection_count": len(rows),
        "invalid_projection_count": len(invalid),
        "ok": not invalid,
        "invalid_projections": invalid,
        "gqa": gqa_audit,
        "projections": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument(
        "--addon-path",
        default=str(PROJECT_ROOT.parents[1] / "AddonOps/build/libTFDLAddOn.so"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    root = Path(args.model_path).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = prefill.QwenPrefillConfig.from_model(root)
    requested_layers = args.layer or list(range(config.num_hidden_layers))
    if len(set(requested_layers)) != len(requested_layers) or any(
        layer < 0 or layer >= config.num_hidden_layers
        for layer in requested_layers
    ):
        raise ValueError("--layer contains a duplicate or invalid layer")
    previous_path = output / "manifest.json"
    previous = (
        json.loads(previous_path.read_text())
        if previous_path.is_file()
        else {}
    )
    previous_artifacts = {
        int(item["layer"]): item
        for item in previous.get("artifacts", [])
        if isinstance(item, dict) and item.get("layer") is not None
    }
    index = prefill.SafeTensorIndex(root)
    artifacts: list[dict[str, object]] = []
    started = time.perf_counter()
    for layer in requested_layers:
        begin = time.perf_counter()
        weights = prefill.load_layer_weights(root, layer, index)
        context, _, inputs, outputs, symbols = prefill.build_layer_graph(
            config,
            layer,
            args.seq_len,
            weights,
            addon_path=args.addon_path,
            fp16_boundaries=True,
            native_masksoftmax=True,
            projection_mode="matmul-aw",
        )
        artifact = output / f"layer_{layer:02d}_seq_{args.seq_len}.fb"
        prefill.dump_context(context, artifact)
        symbol_path = output / f"layer_{layer:02d}.symbols.json"
        symbol_path.write_text(json.dumps(symbols, indent=2, sort_keys=True))
        matmul_audit = _audit_fp16_matmul_projections(
            artifact, symbols, layer, weights, config
        )
        audit_path = output / f"layer_{layer:02d}.matmul-audit.json"
        audit_path.write_text(json.dumps(matmul_audit, indent=2, sort_keys=True))
        if not matmul_audit["ok"]:
            raise RuntimeError(
                f"layer {layer:02d} failed exported FP16 MatMul audit: "
                f"{matmul_audit['invalid_projections']}"
            )
        files = sorted(
            path
            for path in output.glob(artifact.stem + "*")
            if path.is_file()
        )
        artifacts.append(
            {
                "layer": layer,
                "file": artifact.name,
                "files": [
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in files
                ],
                "inputs": inputs,
                "outputs": outputs,
                "symbols": symbol_path.name,
                "matmul_audit": audit_path.name,
                "matmul_audit_ok": True,
                "grouped_gqa_audit_ok": bool(matmul_audit["gqa"]["valid"]),
                "projection_count": matmul_audit["projection_count"],
                "build_seconds": time.perf_counter() - begin,
            }
        )
        print(f"layer {layer:02d}: {artifacts[-1]['build_seconds']:.3f}s -> {artifact}", flush=True)
        del context, weights
        gc.collect()
    for item in artifacts:
        previous_artifacts[int(item["layer"])] = item
    layers = sorted(previous_artifacts)
    manifest = {
        "format": "gelab-zero-4b-prefill-stack-v2",
        "profile": (
            f"gelab-zero-4b-s{args.seq_len}-bsc-fp16-matmul-aw-"
            "grouped-gqa-masksoftmax"
        ),
        "model_path": str(root),
        "seq_len": args.seq_len,
        "layers": layers,
        "precision": "fp16",
        "projection_operator": "MatMul",
        "projection_layout": "A[B,S,K] @ W[K,N]",
        "projection_weight_storage": "checkpoint [N,K] pre-transposed to [K,N]",
        "projection_trans_a": False,
        "projection_trans_b": False,
        "projection_parameter_dtype": "float16",
        "projection_count_per_layer": 7,
        "main_trunk_layout": "BSC",
        "layer_input_layout": "BSC",
        "layer_output_layout": "BSC",
        "attention_internal_layout": "BHSD boundaries; grouped [Hkv,repeat*S,D] MatMul",
        "attention_mode": "fp16-grouped-gqa-qk-scale-masksoftmax-grouped-av",
        "gqa": {
            "query_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "kv_repeats": config.kv_repeats,
            "materialized_kv_repeat": False,
            "repeat_operators": [],
        },
        "softmax_operator": "MaskSoftmax",
        "causal_mask_input": False,
        "residual_layout": "BSC",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "artifact_pattern": f"layer_{{layer:02d}}_seq_{args.seq_len}.fb",
        "artifacts": [previous_artifacts[layer] for layer in layers],
        "total_seconds": time.perf_counter() - started,
        "builder": str(PROJECT_ROOT / "python/qwen_prefill.py"),
    }
    previous_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
