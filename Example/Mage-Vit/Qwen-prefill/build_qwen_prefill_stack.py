#!/usr/bin/env python3
"""Build one fixed-sequence Qwen prefill FB per decoder layer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import qwen_prefill as prefill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--range-json", required=True)
    parser.add_argument(
        "--calibration-report",
        help=(
            "JSON report emitted by collect_qwen_prefill_ranges.py. Its "
            "validated flexible-bucket provenance is copied into manifest.json."
        ),
    )
    parser.add_argument(
        "--calibration-language",
        action="append",
        default=[],
        help="Language represented by calibration prompts; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--outlier-top-k", type=int, default=2)
    parser.add_argument(
        "--attention-mode",
        choices=("arm-causal-hxs", "arm-causal-scalar", "legacy-fp16"),
        default="arm-causal-hxs",
    )
    parser.add_argument(
        "--activation-granularity",
        choices=("scalar", "token"),
        default="scalar",
        help=(
            "token is a diagnostic MatMul experiment that is incompatible "
            "with per-output-channel weights in the current SDK; scalar "
            "uses the production Vit-style Conv1x1 path"
        ),
    )
    parser.add_argument(
        "--per-channel-qk-max-requant-multiplier",
        type=float,
        default=0.99,
    )
    parser.add_argument("--softmax-threads", type=int, default=0)
    parser.add_argument(
        "--skip-tensor-audit",
        action="store_true",
        help=(
            "Skip the final GetAllTensorNames UINT8-qinfo audit. Production "
            "exports should leave this disabled."
        ),
    )
    parser.add_argument(
        "--token-group-boundaries",
        type=int,
        nargs="*",
        default=None,
        metavar="TOKEN",
        help=(
            "Split each non-Top-K MLP at these fixed token offsets so every "
            "prompt role has independent activation qinfo."
        ),
    )
    parser.add_argument(
        "--prompt-dir",
        help=(
            "Prepared prompt used to infer prefix/visual/final-query Tok "
            "hybrid boundaries; mutually exclusive with explicit boundaries."
        ),
    )
    parser.add_argument(
        "--token-hybrid-qkv-start-layer",
        type=int,
        default=None,
        help=(
            "Optionally split Q/K/V token roles from this layer onward; "
            "leave unset for the Vit-style MLP-only baseline."
        ),
    )
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_provenance(
    args: argparse.Namespace,
    layers: list[int],
    output: Path,
) -> dict[str, object] | None:
    """Load explicit provenance, or preserve it during an in-place rebuild."""
    if args.calibration_report:
        report_path = Path(args.calibration_report)
        report = json.loads(report_path.read_text())
        if int(report.get("seq_len", -1)) != args.seq_len:
            raise ValueError(
                f"calibration report seq_len={report.get('seq_len')!r}, "
                f"expected {args.seq_len}"
            )
        report_layers = {int(layer) for layer in report.get("layers", [])}
        missing_layers = sorted(set(layers) - report_layers)
        if missing_layers:
            raise ValueError(
                "calibration report does not cover layers "
                f"{missing_layers}"
            )
        valid_lengths = [
            int(length) for length in report.get("valid_seq_lens", [])
        ]
        if not valid_lengths or any(
            length <= 0 or length > args.seq_len for length in valid_lengths
        ):
            raise ValueError(
                "calibration report has invalid valid_seq_lens: "
                f"{valid_lengths}"
            )
        if args.seq_len not in valid_lengths:
            raise ValueError(
                "calibration must include a full-length prompt so every HxS "
                "QK row is based on real tokens"
            )
        prompts = list(report.get("prompts", []))
        if prompts and len(prompts) != len(valid_lengths):
            raise ValueError(
                "calibration report prompt and valid_seq_lens counts differ"
            )
        for policy in ("padding_ranges_ignored", "causal_qk_cells_only"):
            if policy in report and report[policy] is not True:
                raise ValueError(
                    f"calibration report declares {policy}=false"
                )
        range_path = Path(args.range_json)
        provenance: dict[str, object] = {
            "format": "mage-qwen-prefill-calibration-v1",
            "prompt_count": len(valid_lengths),
            "valid_seq_lens": valid_lengths,
            "languages": list(dict.fromkeys(args.calibration_language)),
            # RangeCollector._valid_values and its QK row collector implement
            # these policies whenever valid_seq_len is supplied.
            "padding_ranges_ignored": True,
            "causal_qk_cells_only": True,
            "range_count": int(report.get("range_count", 0)),
            "row_range_count": int(report.get("row_range_count", 0)),
            "token_range_count": int(report.get("token_range_count", 0)),
            "range_json_sha256": _sha256(range_path),
        }
        return provenance

    # Re-exporting FBs into the same profile must not silently erase the
    # calibration proof already attached to that profile.
    old_manifest = output / "manifest.json"
    if old_manifest.is_file():
        previous = json.loads(old_manifest.read_text()).get("calibration")
        if isinstance(previous, dict):
            valid_lengths = previous.get("valid_seq_lens", [])
            if (
                previous.get("padding_ranges_ignored") is True
                and previous.get("causal_qk_cells_only") is True
                and args.seq_len in valid_lengths
            ):
                return previous
    return None


def main() -> None:
    args = parse_args()
    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    if args.prompt_dir and args.token_group_boundaries is not None:
        raise ValueError(
            "--prompt-dir and --token-group-boundaries are mutually exclusive"
        )
    token_group_boundaries = (
        prefill.infer_token_group_boundaries(args.prompt_dir, args.seq_len)
        if args.prompt_dir
        else tuple(args.token_group_boundaries or ())
    )
    layers = args.layer or list(range(config.num_hidden_layers))
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if len(set(layers)) != len(layers) or any(
        layer < 0 or layer >= config.num_hidden_layers for layer in layers
    ):
        raise ValueError("--layer contains a duplicate or invalid layer")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibration = _calibration_provenance(args, layers, output)
    checkpoint = prefill.SafeTensorIndex(args.model_path)
    reports: list[dict[str, object]] = []
    started = time.perf_counter()
    for layer in layers:
        weights = prefill.load_layer_weights(
            args.model_path, layer, checkpoint
        )
        layer_started = time.perf_counter()
        context, _, inputs, outputs, report = (
            prefill.build_quantlite_int8_layer_graph(
                config,
                layer,
                args.seq_len,
                weights,
                args.range_json,
                top_k=args.outlier_top_k,
                attention_mode=args.attention_mode,
                activation_granularity=args.activation_granularity,
                per_channel_qk_max_requant_multiplier=(
                    args.per_channel_qk_max_requant_multiplier
                ),
                softmax_threads=args.softmax_threads,
                token_group_boundaries=token_group_boundaries,
                token_hybrid_qkv_start_layer=(
                    args.token_hybrid_qkv_start_layer
                ),
                addon_path=args.addon_path,
            )
        )
        artifact = output / f"layer_{layer:02d}_seq_{args.seq_len}.fb"
        prefill.dump_context(context, artifact)
        tensor_audit_summary: dict[str, object] | None = None
        if not args.skip_tensor_audit:
            tensor_audit = prefill.audit_exported_int8_qinfo(
                artifact, args.addon_path
            )
            tensor_audit_path = output / f"layer_{layer:02d}.tensor-audit.json"
            tensor_audit_path.write_text(
                json.dumps(tensor_audit, indent=2, sort_keys=True)
            )
            tensor_audit_summary = {
                "artifact": str(tensor_audit_path),
                "tensor_count": tensor_audit["tensor_count"],
                "dtype_counts": tensor_audit["dtype_counts"],
                "invalid_int8_qinfo_count": tensor_audit[
                    "invalid_int8_qinfo_count"
                ],
                "ok": tensor_audit["ok"],
            }
            if not tensor_audit["ok"]:
                details = ", ".join(
                    f"{item['name']} ({item.get('layer_type')}: "
                    f"{item.get('reason')})"
                    for item in tensor_audit["invalid_int8_qinfo"][:12]
                )
                raise RuntimeError(
                    f"layer {layer:02d} exported UINT8 tensors without "
                    f"valid qinfo: {details}; full report: "
                    f"{tensor_audit_path}"
                )
        symbols = report.pop("symbols")
        symbol_path = output / f"layer_{layer:02d}.symbols.json"
        symbol_path.write_text(json.dumps(symbols, indent=2, sort_keys=True))
        elapsed = time.perf_counter() - layer_started
        report.update(
            {
                "artifact": str(artifact),
                "symbol_map": str(symbol_path),
                "inputs": inputs,
                "outputs": outputs,
                "build_seconds": elapsed,
                "tensor_audit": tensor_audit_summary,
            }
        )
        reports.append(report)
        print(
            f"layer {layer:02d}: {elapsed:.3f}s -> {artifact}",
            flush=True,
        )
        del context, weights
        gc.collect()
    manifest = {
        "format": "mage-qwen-prefill-stack-v1",
        "model_path": args.model_path,
        "seq_len": args.seq_len,
        "layers": layers,
        "attention_mode": args.attention_mode,
        "activation_granularity": args.activation_granularity,
        "outlier_top_k": args.outlier_top_k,
        "softmax_threads": args.softmax_threads,
        "token_group_boundaries": list(token_group_boundaries),
        "token_hybrid_qkv_start_layer": (
            args.token_hybrid_qkv_start_layer
        ),
        "calibration": calibration,
        "range_json": (
            f"build-time-only:{Path(args.range_json).name}"
            if calibration is not None
            else args.range_json
        ),
        "artifact_pattern": f"layer_{{layer:02d}}_seq_{args.seq_len}.fb",
        "total_seconds": time.perf_counter() - started,
        "reports": reports,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
