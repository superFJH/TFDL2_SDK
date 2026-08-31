#!/usr/bin/env python3
"""Check that the selected GELab FP16 deployment is complete and consistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEPLOY_DIR / "deployment.json"))
    parser.add_argument("--model-path")
    parser.add_argument("--allow-missing-model", action="store_true")
    parser.add_argument("--require-vision-accuracy-pass", action="store_true")
    return parser.parse_args()


def _manifest(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.is_file():
        errors.append(f"missing manifest: {path}")
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        errors.append(f"manifest is not an object: {path}")
        return {}
    return value


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = config_path.parent
    errors: list[str] = []
    warnings: list[str] = []
    model = (
        Path(args.model_path).resolve()
        if args.model_path
        else _resolve(root, str(config["model_path"]))
    )
    runtime_assets_root = _resolve(
        root, str(config.get("runtime_assets_dir", "model/runtime"))
    )
    runtime_assets = _manifest(runtime_assets_root / "manifest.json", errors)
    runtime_embedding = runtime_assets.get("token_embedding", {})
    runtime_embedding_path = runtime_assets_root / str(
        runtime_embedding.get("file", "")
    )
    runtime_final_norm = runtime_assets.get("final_norm", {})
    runtime_final_norm_path = runtime_assets_root / str(
        runtime_final_norm.get("file", "")
    )
    if runtime_assets and runtime_assets.get("format") != "gelab-zero-4b-runtime-assets-v1":
        errors.append("runtime assets have an unsupported format")
    if runtime_assets and not (runtime_assets_root / "config.json").is_file():
        errors.append("runtime assets are missing config.json")
    if runtime_assets and not (runtime_assets_root / "tokenizer.json").is_file():
        errors.append("runtime assets are missing tokenizer.json")
    if runtime_assets and runtime_embedding.get("dtype") != "BF16":
        errors.append("runtime token embedding must preserve BF16")
    if runtime_assets and not runtime_embedding_path.is_file():
        errors.append(f"missing runtime token embedding: {runtime_embedding_path}")
    if runtime_assets and runtime_embedding_path.is_file():
        shape = [int(value) for value in runtime_embedding.get("shape", [])]
        if len(shape) != 2 or runtime_embedding_path.stat().st_size != shape[0] * shape[1] * 2:
            errors.append("runtime token embedding shape/byte count is invalid")
    if (
        not args.allow_missing_model
        and not (model / "config.json").is_file()
        and not runtime_assets
    ):
        errors.append(f"missing checkpoint and runtime assets: {model}")
    tfdl = config.get("tfdl")
    if not isinstance(tfdl, dict):
        errors.append("deployment has no unified tfdl configuration")
        tfdl = {}
    for stage in ("vision", "prefill"):
        executor = tfdl.get(stage, {}).get("executor")
        if not isinstance(executor, dict):
            errors.append(f"tfdl.{stage}.executor must be a JSON object")
            continue
        if int(executor.get("cpuLimit", 0)) <= 0:
            errors.append(f"tfdl.{stage}.executor.cpuLimit must be positive")
        if not isinstance(executor.get("optimize"), dict):
            errors.append(f"tfdl.{stage}.executor.optimize must be a JSON object")
    addon = _resolve(root, str(tfdl.get("addon", "")))
    if not addon.is_file():
        errors.append(f"missing addon: {addon}")

    vision_cfg = tfdl.get("vision", {})
    vision_root = _resolve(root, str(vision_cfg["fb_dir"]))
    vision = _manifest(vision_root / "manifest.single.json", errors)
    expected_vision = [
        vision_root / str(item["name"])
        for item in vision.get("files", [])
        if isinstance(item, dict) and item.get("name")
    ]
    for path in expected_vision:
        if not path.is_file():
            errors.append(f"missing vision FB: {path}")
    if vision:
        if vision.get("layout") != "single-logical-graph-external-parameters":
            errors.append("vision is not one logical FB graph")
        if vision.get("precision") != "fp16" or vision.get("quantization") is not None:
            errors.append("vision manifest is not the pure FP16 profile")
        if vision.get("graph_construction") != "direct-TFContext-Vit.py-style":
            errors.append("vision is not built with the Vit.py-style direct graph")
        if vision.get("main_trunk_layout") != "BSC":
            errors.append("vision main trunk is not BSC")
        if vision.get("projection_input_layout") != "A[B,S,K]":
            errors.append("vision projection activation layout is not A[B,S,K]")
        if vision.get("projection_layout") != "A[B,S,K] @ W[K,N]":
            errors.append("vision projection layout is not activation-left MatMul")
        if vision.get("projection_trans_a") is not False:
            errors.append("vision projection transA must be false")
        if vision.get("projection_trans_b") is not False:
            errors.append("vision projection transB must be false")
        topology_audit = vision.get("topology_audit", {})
        if not topology_audit.get("ok"):
            errors.append("vision BSC MatMul topology audit did not pass")
        if topology_audit.get("covered_transformer_logical_projections") != 144:
            errors.append("vision topology audit does not cover all 144 block projections")
        if topology_audit.get("projection_operator") != "MatMul":
            errors.append("vision topology audit does not use MatMul projections")
        if topology_audit.get("projection_matmul_nodes") != 153:
            errors.append("vision topology audit does not cover all 153 former Conv1x1 projections")
        if topology_audit.get("convolution_projection_nodes") != 0:
            errors.append("vision still has a Convolution projection")
        if topology_audit.get("projection_trans_a") is not False:
            errors.append("vision topology audit has transA projection")
        if topology_audit.get("projection_trans_b") is not False:
            errors.append("vision topology audit has transB projection")
        diagnostics = vision.get("diagnostics", {})
        for key in ("symbols", "topology_audit"):
            diagnostic = vision_root / str(diagnostics.get(key, ""))
            if not diagnostic.is_file():
                errors.append(f"missing vision {key} diagnostic: {diagnostic}")
        if vision.get("parameterized_linear_lowering") != "MatMul":
            errors.append("vision parameterized Linear operators are not MatMul")
        if (
            vision.get("grid_h"),
            vision.get("grid_w"),
        ) != (vision_cfg.get("grid_h"), vision_cfg.get("grid_w")):
            errors.append(
                "vision FB grid does not match deployment preprocessing grid"
            )
        if len(vision.get("inputs", [])) != 1:
            errors.append("fixed-grid single vision FB must expose only the pixel-patch input")
        topology = [
            path
            for path in expected_vision
            if path.name.endswith(".fb") and not path.name.endswith(".param.fb")
        ]
        if len(topology) != 1:
            errors.append(f"single vision must have exactly one topology FB, got {topology}")

    prefill_cfg = tfdl.get("prefill", {})
    if int(prefill_cfg.get("prefill_chunk_layers", -1)) < 0:
        errors.append("tfdl.prefill.prefill_chunk_layers must be non-negative")
    prefill_root = _resolve(root, str(prefill_cfg["fb_dir"]))
    prefill = _manifest(prefill_root / "manifest.json", errors)
    expected_prefill = [
        prefill_root / f"layer_{layer:02d}_seq_{prefill_cfg['seq_len']}.fb"
        for layer in range(36)
    ]
    for path in expected_prefill:
        if not path.is_file():
            errors.append(f"missing prefill FB: {path}")
    if prefill:
        if prefill.get("precision") != "fp16":
            errors.append("prefill manifest is not FP16")
        if prefill.get("main_trunk_layout") != "BSC":
            errors.append("prefill main trunk is not BSC")
        if prefill.get("projection_operator") != "MatMul":
            errors.append("prefill projections are not MatMul")
        if prefill.get("projection_layout") != "A[B,S,K] @ W[K,N]":
            errors.append("prefill projection inputs are not activation-left/weight-right")
        if prefill.get("projection_trans_a") is not False:
            errors.append("prefill projection transA must be false")
        if prefill.get("projection_trans_b") is not False:
            errors.append("prefill projection transB must be false")
        if prefill.get("projection_parameter_dtype") != "float16":
            errors.append("prefill projection weights are not FP16")
        if prefill.get("softmax_operator") != "MaskSoftmax":
            errors.append("prefill does not use native MaskSoftmax")
        if prefill.get("attention_mode") != (
            "fp16-grouped-gqa-qk-scale-masksoftmax-grouped-av"
        ):
            errors.append("prefill is not the grouped-GQA attention profile")
        gqa = prefill.get("gqa", {})
        if gqa != {
            "query_heads": 32,
            "kv_heads": 8,
            "kv_repeats": 4,
            "materialized_kv_repeat": False,
            "repeat_operators": [],
        }:
            errors.append("prefill GQA contract is not 32Q/8KV without K/V repeat")
        if prefill.get("layers") != list(range(36)):
            errors.append("prefill manifest does not contain ordered layers 0..35")
        artifacts = {
            int(item["layer"]): item
            for item in prefill.get("artifacts", [])
            if isinstance(item, dict) and item.get("layer") is not None
        }
        for layer in range(36):
            item = artifacts.get(layer, {})
            audit_name = item.get("matmul_audit")
            if not audit_name:
                errors.append(f"prefill layer {layer:02d} has no MatMul audit")
                continue
            audit = _manifest(prefill_root / str(audit_name), errors)
            if not audit.get("ok") or audit.get("projection_count") != 7:
                errors.append(
                    f"prefill layer {layer:02d} failed its 7-projection MatMul audit"
                )
            if not audit.get("gqa", {}).get("valid"):
                errors.append(
                    f"prefill layer {layer:02d} failed its grouped-GQA audit"
                )

    decode_cfg = config["decode"]
    decode_engine = str(decode_cfg.get("engine", "llama.cpp-external-kv"))
    if decode_engine != "llama.cpp-external-kv":
        errors.append(f"unsupported fixed decode engine: {decode_engine}")
    if runtime_final_norm.get("dtype") != "BF16":
        errors.append("llama.cpp decoder requires BF16 runtime final RMSNorm")
    embedding_shape = [int(value) for value in runtime_embedding.get("shape", [])]
    norm_shape = [int(value) for value in runtime_final_norm.get("shape", [])]
    if (
        len(embedding_shape) != 2
        or norm_shape != [embedding_shape[1]]
        or not runtime_final_norm_path.is_file()
        or runtime_final_norm_path.stat().st_size != embedding_shape[1] * 2
    ):
        errors.append("runtime final RMSNorm shape/byte count is invalid")
    decode_root = _resolve(root, str(decode_cfg["decoder_dir"]))
    decode = _manifest(decode_root / "manifest.json", errors)
    expected_decode: list[Path] = []
    if decode:
        expected_decode.append(decode_root / str(decode.get("model_file", "")))
    for path in expected_decode:
        if not path.is_file():
            errors.append(f"missing decode artifact: {path}")
    if decode:
        if decode.get("format") != "tfdl-llmdecode-gguf-v1":
            errors.append("decoder is not the tfdl-llmdecode GGUF format")
        if decode.get("cache_dtype") != "float16":
            errors.append("llama.cpp external KV cache source must be FP16")
        if not decode.get("tie_word_embeddings"):
            errors.append("checkpoint-free tied final head is not enabled")
        if not str(decode_cfg.get("valid_kv_length", "")):
            errors.append("llama.cpp decoder must declare dynamic valid_kv_length")

    accuracy_path = DEPLOY_DIR / "FP16_ACCURACY_REPORT.json"
    accuracy = _manifest(accuracy_path, errors if args.require_vision_accuracy_pass else [])
    accuracy_status = accuracy.get("status") if accuracy else None
    reported_artifact = (
        accuracy.get("vision", {}).get("artifact", {}).get("file")
        if isinstance(accuracy, dict)
        else None
    )
    selected_artifact = (
        str(Path("model/vision") / Path(str(vision_cfg.get("fb_dir", ""))).name / expected_vision[0].name)
        if expected_vision
        else None
    )
    if reported_artifact != selected_artifact:
        accuracy_status = "unmeasured"
    if accuracy_status != "pass":
        message = (
            f"FP16 vision accuracy status is {accuracy_status or 'unmeasured'}: {accuracy_path}"
        )
        if reported_artifact != selected_artifact:
            message += (
                f" (report artifact {reported_artifact!r} does not match selected "
                f"artifact {selected_artifact!r})"
            )
        if args.require_vision_accuracy_pass:
            errors.append(message)
        else:
            warnings.append(message)

    report = {
        "ok": not errors,
        "profile": config["profile"],
        "checkpoint": str(model),
        "runtime_assets": str(runtime_assets_root),
        "runtime_assets_available": bool(runtime_assets),
        "vision_fb_count": sum(path.is_file() for path in expected_vision),
        "expected_vision_fb_count": len(expected_vision),
        "prefill_fb_count": sum(path.is_file() for path in expected_prefill),
        "expected_prefill_fb_count": len(expected_prefill),
        "prefill_projection_audit_count": sum(
            (prefill_root / f"layer_{layer:02d}.matmul-audit.json").is_file()
            for layer in range(36)
        ),
        "decode_file_count": sum(path.is_file() for path in expected_decode),
        "expected_decode_file_count": len(expected_decode),
        "decode_engine": decode_engine,
        "accuracy_status": accuracy_status,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
