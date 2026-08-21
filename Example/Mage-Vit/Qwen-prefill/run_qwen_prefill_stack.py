#!/usr/bin/env python3
"""Run fixed-sequence Qwen layer FBs, emit KV cache, and align logits."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fb-dir", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--executor-lifetime",
        choices=("auto", "per-layer", "retain-contexts", "retain-all"),
        default="auto",
        help=(
            "Control native object lifetime. auto retains contexts on NPU "
            "and releases them per layer in software. "
            "retain-contexts keeps every shared-weight TFContext alive; "
            "retain-all also keeps every compiled TFExecutor alive."
        ),
    )
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument(
        "--reference-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--final-device", default="cpu")
    parser.add_argument("--lm-head-block-rows", type=int, default=4096)
    parser.add_argument("--no-final-logits", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--require-token-group-match",
        action="store_true",
        help=(
            "fail before execution when this prompt's image-token boundaries "
            "differ from the static Tok-hybrid boundaries baked into the FBs"
        ),
    )
    parser.add_argument(
        "--check-prompt-only",
        action="store_true",
        help="report prompt/profile compatibility without loading TFDL or FBs",
    )
    parser.add_argument(
        "--addon-path",
        "--addon",
        dest="addon_path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return parser.parse_args()


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(actual, dtype=np.float64).reshape(-1)
    delta = right - left
    norm = float(np.linalg.norm(left))
    denominator = norm * float(np.linalg.norm(right))
    return {
        "cosine": float(np.dot(left, right) / denominator)
        if denominator else 0.0,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rel_l2": float(np.linalg.norm(delta) / norm) if norm else 0.0,
    }


def sequence_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    sequence_axis: int,
    text_mask: np.ndarray | None = None,
) -> dict[str, object]:
    left = np.moveaxis(np.asarray(reference, dtype=np.float64), sequence_axis, 0)
    right = np.moveaxis(np.asarray(actual, dtype=np.float64), sequence_axis, 0)
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine_per_token = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.zeros(left.shape[0], dtype=np.float64),
        where=denominator != 0.0,
    )
    result: dict[str, object] = {
        "last_token": metrics(left[-1], right[-1]),
        "token_cosine": {
            "minimum": float(cosine_per_token.min()),
            "p01": float(np.percentile(cosine_per_token, 1)),
            "p10": float(np.percentile(cosine_per_token, 10)),
            "median": float(np.median(cosine_per_token)),
            "mean": float(cosine_per_token.mean()),
        },
    }
    if text_mask is not None:
        mask = np.asarray(text_mask, dtype=bool).reshape(-1)
        if mask.shape != (left.shape[0],):
            raise ValueError(
                f"text mask {mask.shape}, expected {(left.shape[0],)}"
            )
        result["text"] = metrics(left[mask], right[mask])
        result["visual"] = metrics(left[~mask], right[~mask])
    return result


def _load_prompt(
    directory: Path, config: prefill.QwenPrefillConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    hidden = np.load(directory / "hidden.npy").astype(np.float32)
    positions = np.load(directory / "position_ids.npy").reshape(-1)
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"invalid prompt hidden shape {hidden.shape}")
    expected = (1, positions.size, config.hidden_size)
    if hidden.shape != expected:
        raise ValueError(f"prompt hidden {hidden.shape}, expected {expected}")
    metadata_path = directory / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists() else {}
    )
    return hidden, positions.astype(np.int64), metadata


def _valid_sequence_length(directory: Path, sequence: int, metadata: dict[str, object]) -> int:
    """Validate the one real prefix followed by optional right-padding."""
    mask_path = directory / "attention_mask.npy"
    mask = np.load(mask_path).reshape(-1) if mask_path.exists() else None
    inferred = int(mask.sum()) if mask is not None else sequence
    valid = int(metadata.get("valid_seq_len", inferred))
    if not 0 < valid <= sequence:
        raise ValueError(f"valid_seq_len={valid}, expected inside [1, {sequence}]")
    if int(metadata.get("model_seq_len", sequence)) != sequence:
        raise ValueError("prompt metadata model_seq_len does not match tensors")
    if mask is not None:
        expected = np.zeros(sequence, dtype=mask.dtype)
        expected[:valid] = 1
        if mask.shape != (sequence,) or not np.array_equal(mask, expected):
            raise ValueError(
                "attention_mask must contain one valid prefix followed by "
                "right-padding"
            )
    return valid


def _token_group_compatibility(
    fb_dir: Path,
    prompt_dir: Path,
    prompt_metadata: dict[str, object],
    valid_sequence: int,
) -> dict[str, object]:
    """Compare runtime prompt roles with static Tok-hybrid graph slices."""
    manifest_path = fb_dir / "manifest.json"
    model_boundaries: tuple[int, ...] = ()
    calibration_lengths: list[int] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        model_boundaries = tuple(
            int(value) for value in manifest.get("token_group_boundaries", [])
        )
        calibration = manifest.get("calibration", {})
        if isinstance(calibration, dict):
            calibration_lengths = [
                int(value) for value in calibration.get("valid_seq_lens", [])
            ]
    input_ids_path = prompt_dir / "input_ids.npy"
    actual_boundaries: tuple[int, ...] = ()
    reason = "profile has no Tok-hybrid boundaries"
    compatible = True
    if model_boundaries:
        if not input_ids_path.is_file() or "image_token_id" not in prompt_metadata:
            compatible = False
            reason = "prompt lacks input_ids.npy or image_token_id"
        else:
            actual_boundaries = prefill.prompt_token_group_boundaries(
                np.load(input_ids_path),
                int(prompt_metadata["image_token_id"]),
                valid_seq_len=valid_sequence,
            )
            # A coarser profile such as prefix-versus-rest `[21]` is valid
            # for an actual `[21,770]` prompt. Every split the graph does bake
            # in must still coincide with a real prompt-role boundary.
            compatible = all(
                value in actual_boundaries for value in model_boundaries
            )
            reason = (
                "all static graph splits align with prompt-role boundaries"
                if compatible
                else (
                    "static Tok-hybrid Slice/Concat groups do not match this "
                    "video prompt"
                )
            )
    inside_calibrated_length_span = (
        min(calibration_lengths) <= valid_sequence <= max(calibration_lengths)
        if calibration_lengths else None
    )
    if inside_calibrated_length_span is False:
        compatible = False
        reason += "; valid_seq_len is outside the calibration length span"
    return {
        "compatible": compatible,
        "model_boundaries": list(model_boundaries),
        "actual_boundaries": list(actual_boundaries),
        "valid_seq_len": valid_sequence,
        "calibration_valid_seq_lens": calibration_lengths,
        "inside_calibrated_length_span": inside_calibrated_length_span,
        "reason": reason,
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
    }


def main() -> None:
    args = parse_args()
    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    prompt_root = Path(args.prompt_dir)
    hidden, positions, prompt_metadata = _load_prompt(prompt_root, config)
    sequence = positions.size
    valid_sequence = _valid_sequence_length(
        prompt_root, sequence, prompt_metadata
    )
    token_group_compatibility = _token_group_compatibility(
        Path(args.fb_dir), prompt_root, prompt_metadata, valid_sequence
    )
    print(
        "token_group_compatibility="
        + json.dumps(token_group_compatibility, ensure_ascii=False),
        flush=True,
    )
    if not token_group_compatibility["compatible"]:
        message = (
            f"prefill profile is incompatible ({token_group_compatibility['reason']}): "
            f"actual_boundaries={token_group_compatibility['actual_boundaries']}, "
            f"model_boundaries={token_group_compatibility['model_boundaries']}, "
            f"valid_seq_len={valid_sequence}"
        )
        if args.require_token_group_match:
            raise ValueError(message)
        print(f"WARNING: {message}; accuracy is not validated", flush=True)
    if token_group_compatibility["inside_calibrated_length_span"] is False:
        print(
            "WARNING: valid_seq_len is outside the calibration length span; "
            "fixed activation qinfo is extrapolating to this prompt",
            flush=True,
        )
    if args.check_prompt_only:
        return
    layers = args.layer or list(range(config.num_hidden_layers))
    if len(set(layers)) != len(layers) or any(
        layer < 0 or layer >= config.num_hidden_layers for layer in layers
    ):
        raise ValueError("--layer contains a duplicate or invalid layer")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sin, cos = prefill.compute_rope(
        positions, config.head_dim, config.rope_theta
    )
    _, TFContext, TFExecutor, _ = prefill._load_tfdl(args.addon_path)
    checkpoint = prefill.SafeTensorIndex(args.model_path)
    input_ids_path = prompt_root / "input_ids.npy"
    text_mask = None
    if input_ids_path.exists() and "image_token_id" in prompt_metadata:
        input_ids = np.load(input_ids_path).reshape(-1)
        text_mask = (
            input_ids[:valid_sequence]
            != int(prompt_metadata["image_token_id"])
        )

    reference_hidden = None
    torch_dtype = None
    torch_device = None
    if args.compare_reference:
        import torch

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.reference_dtype]
        torch_device = torch.device(args.reference_device)
        reference_hidden = torch.from_numpy(hidden).to(
            device=torch_device, dtype=torch_dtype
        )
        reference_sin = torch.from_numpy(sin).to(
            device=torch_device, dtype=torch_dtype
        )
        reference_cos = torch.from_numpy(cos).to(
            device=torch_device, dtype=torch_dtype
        )

    reports: list[dict[str, object]] = []
    retained_contexts: list[object] = []
    retained_executors: list[object] = []
    executor_lifetime = args.executor_lifetime
    if executor_lifetime == "auto":
        # On NPU40T, destroying a previously executed shared-weight context
        # during a multi-FB stack invalidates later hardware state.  Defer all
        # context destruction until the complete prefill has finished.  The
        # software executor does not need this memory-heavy workaround.
        executor_lifetime = (
            "retain-contexts" if args.hardware else "per-layer"
        )
    print(
        f"executor_lifetime={executor_lifetime} "
        f"(requested={args.executor_lifetime})",
        flush=True,
    )
    total_started = time.perf_counter()
    for layer in layers:
        artifact = (
            Path(args.fb_dir) / f"layer_{layer:02d}_seq_{sequence}.fb"
        )
        if not artifact.exists():
            raise FileNotFoundError(f"missing layer artifact {artifact}")
        compile_started = time.perf_counter()
        # TFExecutor compiles with shareWeight=true.  Keep this context alive
        # until after the executor is destroyed; a temporary TFContext here
        # leaves hardware weight buffers dangling and can corrupt later layers.
        context = TFContext(path=str(artifact))
        executor = TFExecutor(
            context,
            prefill.prefill_executor_config(
                bool(args.hardware),
                software_attn_softmax_impl=False,
            ),
        )
        compile_seconds = time.perf_counter() - compile_started
        execute_started = time.perf_counter()
        hidden_next, key, value = prefill.execute_layer(
            executor, hidden, sin, cos
        )
        execute_seconds = time.perf_counter() - execute_started
        # Padding rows are an implementation detail of the fixed NPU graph.
        # The CPU decoder must receive a dense cache containing real tokens
        # only, or its next position and causal length would be wrong.
        key_valid = np.ascontiguousarray(key[:, :, :valid_sequence])
        value_valid = np.ascontiguousarray(value[:, :, :valid_sequence])
        np.save(output / f"layer_{layer:02d}.key.npy", key_valid)
        np.save(output / f"layer_{layer:02d}.value.npy", value_valid)
        layer_report: dict[str, object] = {
            "layer": layer,
            "artifact": str(artifact),
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
            "hidden_dtype": str(hidden_next.dtype),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
        }
        if reference_hidden is not None:
            import torch

            weights = prefill.load_layer_weights(
                args.model_path, layer, checkpoint
            )
            with torch.inference_mode():
                reference_hidden, reference_key, reference_value = (
                    prefill.torch_layer(
                        config,
                        layer,
                        weights,
                        reference_hidden,
                        reference_sin,
                        reference_cos,
                    )
                )
            reference_arrays = (
                reference_hidden.detach().float().cpu().numpy(),
                reference_key.detach().float().cpu().numpy(),
                reference_value.detach().float().cpu().numpy(),
            )
            layer_report["comparisons"] = {
                label: metrics(reference, actual)
                for label, reference, actual in zip(
                    ("hidden", "key", "value"),
                    (
                        reference_arrays[0][:, :valid_sequence],
                        reference_arrays[1][:, :, :valid_sequence],
                        reference_arrays[2][:, :, :valid_sequence],
                    ),
                    (
                        hidden_next[:, :valid_sequence],
                        key_valid,
                        value_valid,
                    ),
                )
            }
            layer_report["sequence_comparisons"] = {
                "hidden": sequence_metrics(
                    reference_arrays[0][:, :valid_sequence],
                    hidden_next[:, :valid_sequence],
                    sequence_axis=1, text_mask=text_mask,
                ),
                "key": sequence_metrics(
                    reference_arrays[1][:, :, :valid_sequence],
                    key_valid,
                    sequence_axis=2,
                ),
                "value": sequence_metrics(
                    reference_arrays[2][:, :, :valid_sequence],
                    value_valid,
                    sequence_axis=2,
                ),
            }
            del weights, reference_key, reference_value
        hidden = np.ascontiguousarray(hidden_next)
        reports.append(layer_report)
        comparison = layer_report.get("comparisons", {})
        sequence_comparison = layer_report.get("sequence_comparisons", {})
        last_hidden_cosine = None
        if isinstance(sequence_comparison, dict):
            hidden_sequence = sequence_comparison.get("hidden", {})
            if isinstance(hidden_sequence, dict):
                last_token = hidden_sequence.get("last_token", {})
                if isinstance(last_token, dict):
                    last_hidden_cosine = last_token.get("cosine")
        cosine_values = (
            {
                name: comparison.get(name, {}).get("cosine")
                for name in ("hidden", "key", "value")
            }
            if isinstance(comparison, dict) else {}
        )
        suffix = "" if not cosine_values else (
            ", " + " ".join(
                f"{name}_cos={value:.8f}"
                for name, value in cosine_values.items()
                if value is not None
            )
        )
        if last_hidden_cosine is not None:
            suffix += f" last_hidden_cos={float(last_hidden_cosine):.8f}"
        print(
            f"layer {layer:02d}: compile={compile_seconds:.3f}s "
            f"execute={execute_seconds:.3f}s{suffix}",
            flush=True,
        )
        if executor_lifetime == "retain-all":
            retained_contexts.append(context)
            retained_executors.append(executor)
        elif executor_lifetime == "retain-contexts":
            retained_contexts.append(context)
            del executor
        else:
            del executor
            del context
        gc.collect()

    np.save(
        output / "final_hidden.npy",
        np.ascontiguousarray(hidden[:, :valid_sequence]),
    )
    complete_stack = layers == list(range(config.num_hidden_layers))
    logits_report: dict[str, object] | None = None
    if not args.no_final_logits:
        if not complete_stack:
            raise ValueError(
                "final logits require all layers in order; pass "
                "--no-final-logits for a partial stack"
            )
        logits_started = time.perf_counter()
        logits = prefill.compute_final_logits(
            args.model_path,
            hidden[:, valid_sequence - 1 : valid_sequence],
            config,
            device=args.final_device,
            block_rows=args.lm_head_block_rows,
            index=checkpoint,
        )
        logits_seconds = time.perf_counter() - logits_started
        np.save(output / "last_token_logits.npy", logits)
        top10 = np.argsort(logits)[-10:][::-1]
        logits_report = {
            "seconds": logits_seconds,
            "top1": int(top10[0]),
            "top10": [int(value) for value in top10],
        }
        if reference_hidden is not None:
            reference_logits = prefill.compute_final_logits(
                args.model_path,
                reference_hidden[
                    :, valid_sequence - 1 : valid_sequence
                ].detach().float().cpu().numpy(),
                config,
                device=args.final_device,
                block_rows=args.lm_head_block_rows,
                index=checkpoint,
            )
            reference_top10 = np.argsort(reference_logits)[-10:][::-1]
            logits_report.update(
                {
                    "comparison": metrics(reference_logits, logits),
                    "reference_top1": int(reference_top10[0]),
                    "top1_agreement": bool(top10[0] == reference_top10[0]),
                    "top10_overlap": int(
                        len(set(top10.tolist()) & set(reference_top10.tolist()))
                    ),
                    "reference_top10": [
                        int(value) for value in reference_top10
                    ],
                }
            )

    manifest = {
        "format": "mage-qwen-prefill-kv-v1",
        "model_path": args.model_path,
        "prompt": prompt_metadata,
        "token_group_compatibility": token_group_compatibility,
        # seq_len is the exported KV length for the established decoder ABI.
        "seq_len": valid_sequence,
        "valid_seq_len": valid_sequence,
        "model_seq_len": sequence,
        "layers": layers,
        "hidden_size": config.hidden_size,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "cache_dtype": "float16",
        "cache_files": {
            str(layer): {
                "key": f"layer_{layer:02d}.key.npy",
                "value": f"layer_{layer:02d}.value.npy",
            }
            for layer in layers
        },
        "final_hidden": "final_hidden.npy",
        "logits": logits_report,
        "use_hardware": bool(args.hardware),
        "executor_lifetime": executor_lifetime,
        "requested_executor_lifetime": args.executor_lifetime,
        "total_seconds": time.perf_counter() - total_started,
        "layer_reports": reports,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report_path = Path(args.output_json) if args.output_json else (
        output / "report.json"
    )
    report_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "seq_len": valid_sequence,
        "model_seq_len": sequence,
        "layers": layers,
        "total_seconds": manifest["total_seconds"],
        "logits": logits_report,
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
