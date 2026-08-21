#!/usr/bin/env python3
"""Build and publish the model artifacts used by the Mage-Vit demo.

The expensive builders are kept in their original directories.  This file is
the deployment-facing orchestrator: it validates calibration inputs, builds
into a staging directory, checks the resulting package, and only then replaces
the selected directories below deploy/models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEPLOY_DIR = Path(__file__).resolve().parent
MAGE_ROOT = DEPLOY_DIR.parent
SDK_ROOT = MAGE_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = DEPLOY_DIR / "models"
DEFAULT_WORK_DIR = DEPLOY_DIR / "var" / "model-export"
DEFAULT_ADDON = DEPLOY_DIR / "runtime" / "libTFDLAddOn.so"
VISION_NAME = "mage_vit_288x512.int8_fp16_topk2.fb"
PREFILL_TOPKS = (4, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def profile_name(sequence: int, top_k: int, qkv_start: int) -> str:
    return f"s{sequence}-flex-topk{top_k}-qkv{qkv_start}"


def _nested(config: dict[str, Any], group: str, name: str) -> Any:
    section = config.get(group)
    return section.get(name) if isinstance(section, dict) else None


def validate_checkpoint(model_path: Path) -> dict[str, Any]:
    """Reject checkpoints that do not match the deployed Mage-VL ABI."""
    config_path = model_path / "config.json"
    config = load_json(config_path)
    required = {
        ("vision_config", "hidden_size"): 1024,
        ("vision_config", "intermediate_size"): 4096,
        ("vision_config", "num_hidden_layers"): 24,
        ("vision_config", "num_attention_heads"): 16,
        ("vision_config", "patch_size"): 16,
        ("vision_config", "spatial_merge_size"): 2,
        ("vision_config", "out_hidden_size"): 2560,
        ("text_config", "hidden_size"): 2560,
        ("text_config", "intermediate_size"): 9728,
        ("text_config", "num_hidden_layers"): 36,
        ("text_config", "num_attention_heads"): 32,
        ("text_config", "num_key_value_heads"): 8,
        ("text_config", "head_dim"): 128,
    }
    errors = []
    for (group, name), expected in required.items():
        actual = _nested(config, group, name)
        if actual != expected:
            errors.append(f"{group}.{name}={actual!r}, expected {expected!r}")
    if config.get("model_type") != "mage_vl":
        errors.append(
            f"model_type={config.get('model_type')!r}, expected 'mage_vl'"
        )
    weight_index = model_path / "model.safetensors.index.json"
    single_weights = model_path / "model.safetensors"
    if not weight_index.is_file() and not single_weights.is_file():
        errors.append("model.safetensors.index.json/model.safetensors is absent")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        if not (model_path / name).is_file():
            errors.append(f"{name} is absent")
    if errors:
        raise ValueError(
            f"{model_path} is not compatible with this deployment ABI:\n  - "
            + "\n  - ".join(errors)
        )
    fingerprint = {"config_sha256": sha256(config_path)}
    if weight_index.is_file():
        fingerprint["weight_index_sha256"] = sha256(weight_index)
    else:
        fingerprint["single_weight_size"] = single_weights.stat().st_size
    fingerprint["tokenizer_config_sha256"] = sha256(
        model_path / "tokenizer_config.json"
    )
    fingerprint["tokenizer_json_sha256"] = sha256(model_path / "tokenizer.json")
    return {"config": config, "fingerprint": fingerprint}


def validate_vision_bundles(values: Iterable[str], max_calib: int) -> list[Path]:
    bundles = [Path(value).resolve() for value in values]
    if not bundles:
        raise ValueError("vision export requires --vision-calibration-bundle")
    total = 0
    for root in bundles:
        manifest = load_json(root / "manifest.json")
        height = int(manifest.get("canvas_height", -1))
        width = int(manifest.get("canvas_width", -1))
        if (height, width) != (288, 512):
            raise ValueError(
                f"{root}: canvas is {height}x{width}, expected 288x512"
            )
        canvases = manifest.get("canvases")
        if not isinstance(canvases, list) or not canvases:
            raise ValueError(f"{root}: manifest has no canvases")
        selected = canvases[:max_calib] if max_calib > 0 else canvases
        for entry in selected:
            image = root / str(entry.get("file", ""))
            if not image.is_file():
                raise ValueError(f"{root}: missing calibration canvas {image.name}")
        total += len(selected)
    print(f"validated {len(bundles)} vision bundles / {total} canvases")
    return bundles


def validate_prompt_dirs(
    values: Iterable[str],
    sequence: int,
    hidden_size: int,
    image_token_id: int,
    boundaries: tuple[int, ...],
) -> tuple[list[Path], list[int], list[str]]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required to validate prefill prompts") from error

    prompts = [Path(value).resolve() for value in values]
    if not prompts:
        raise ValueError("prefill export requires --prefill-prompt-dir")
    valid_lengths: list[int] = []
    systems: list[str] = []
    first_image_offsets: list[int] = []
    for root in prompts:
        metadata = load_json(root / "metadata.json")
        hidden_path = root / "hidden.npy"
        position_path = root / "position_ids.npy"
        input_ids_path = root / "input_ids.npy"
        mask_path = root / "attention_mask.npy"
        for path in (hidden_path, position_path, input_ids_path, mask_path):
            if not path.is_file():
                raise ValueError(f"{root}: missing {path.name}")
        hidden = np.load(hidden_path, mmap_mode="r")
        positions = np.load(position_path, mmap_mode="r")
        input_ids = np.load(input_ids_path, mmap_mode="r")
        attention_mask = np.load(mask_path, mmap_mode="r")
        if hidden.shape != (1, sequence, hidden_size):
            raise ValueError(
                f"{root}: hidden shape {hidden.shape}, expected "
                f"{(1, sequence, hidden_size)}"
            )
        if positions.size != sequence or input_ids.shape != (1, sequence):
            raise ValueError(f"{root}: position/input ID shape is not S={sequence}")
        if attention_mask.shape != (1, sequence):
            raise ValueError(f"{root}: attention mask shape is not S={sequence}")
        valid = int(metadata.get("valid_seq_len", int(attention_mask.sum())))
        if not 0 < valid <= sequence:
            raise ValueError(f"{root}: invalid valid_seq_len={valid}")
        if int(attention_mask[0, :valid].sum()) != valid or int(
            attention_mask[0, valid:].sum()
        ) != 0:
            raise ValueError(f"{root}: prompt must use contiguous right padding")
        image_positions = np.flatnonzero(input_ids[0, :valid] == image_token_id)
        if image_positions.size == 0:
            raise ValueError(f"{root}: prompt contains no image tokens")
        first_image_offsets.append(int(image_positions[0]))
        valid_lengths.append(valid)
        systems.append(str(metadata.get("system", "")))
    if sequence not in valid_lengths:
        raise ValueError(
            f"prefill calibration lengths are {valid_lengths}; include one real "
            f"valid_seq_len={sequence} prompt so every HxS row is calibrated"
        )
    if len(set(systems)) != 1:
        raise ValueError("all prefill calibration prompts must use one system prompt")
    if len(boundaries) == 1 and any(
        offset != boundaries[0] for offset in first_image_offsets
    ):
        raise ValueError(
            f"first image-token offsets are {first_image_offsets}, but the fixed "
            f"Tok-hybrid prefix boundary is {boundaries[0]}; rebuild prompts or "
            "pass the matching --token-group-boundaries value"
        )
    print(
        f"validated {len(prompts)} prefill prompts; valid lengths="
        f"{valid_lengths}, first image token={first_image_offsets}"
    )
    return prompts, valid_lengths, systems


class CommandRunner:
    def __init__(self, dry_run: bool, log_path: Path):
        self.dry_run = dry_run
        self.log_path = log_path
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> None:
        command = [str(value) for value in command]
        self.commands.append(command)
        rendered = shlex.join(command)
        print(f"\n+ {rendered}", flush=True)
        if self.dry_run:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n+ {rendered}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=SDK_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            result = process.wait()
        if result:
            raise RuntimeError(
                f"command failed with return code {result}: {rendered}; "
                f"full log: {self.log_path}"
            )


def _replace_json_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_json_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_json_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def rewrite_staged_paths(stage: Path, destination: Path) -> None:
    old = str(stage.resolve())
    new = str(destination.resolve())
    for path in stage.rglob("*.json"):
        value = load_json(path)
        rewritten = _replace_json_strings(value, old, new)
        if rewritten != value:
            write_json_atomic(path, rewritten)


def audit_vision_fb(fb_path: Path, addon_path: Path, output: Path) -> None:
    qwen_dir = MAGE_ROOT / "Qwen-prefill"
    sys.path.insert(0, str(qwen_dir))
    try:
        import qwen_prefill

        report = qwen_prefill.audit_exported_int8_qinfo(fb_path, addon_path)
    finally:
        try:
            sys.path.remove(str(qwen_dir))
        except ValueError:
            pass
    write_json_atomic(output, report)
    if not report.get("ok"):
        invalid = report.get("invalid_int8_qinfo", [])
        preview = ", ".join(str(row.get("name")) for row in invalid[:8])
        raise RuntimeError(
            f"vision FB contains UINT8 tensors without valid qinfo: {preview}; "
            f"report: {output}"
        )


def validate_vision_output(root: Path, *, require_audit: bool = True) -> None:
    for name in (VISION_NAME, "build.report.json", "symbols.json"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"vision export is incomplete: {path}")
    audit_path = root / "tensor-audit.json"
    if require_audit and not audit_path.is_file():
        raise RuntimeError(f"vision export is missing its qinfo audit: {audit_path}")
    if audit_path.is_file():
        audit = load_json(audit_path)
        if audit.get("ok") is not True:
            raise RuntimeError("vision tensor qinfo audit did not pass")


def validate_prefill_output(
    root: Path,
    sequence: int,
    num_layers: int,
    top_k: int,
    boundaries: tuple[int, ...],
    qkv_start: int,
) -> None:
    fbs = sorted(root.glob(f"layer_*_seq_{sequence}.fb"))
    audits = sorted(root.glob("layer_*.tensor-audit.json"))
    symbols = sorted(root.glob("layer_*.symbols.json"))
    if len(fbs) != num_layers or len(audits) != num_layers or len(symbols) != num_layers:
        raise RuntimeError(
            f"{root}: expected {num_layers} FB/audit/symbol files, got "
            f"{len(fbs)}/{len(audits)}/{len(symbols)}"
        )
    for path in audits:
        audit = load_json(path)
        if audit.get("ok") is not True or int(
            audit.get("invalid_int8_qinfo_count", -1)
        ) != 0:
            raise RuntimeError(f"prefill qinfo audit failed: {path}")
    manifest = load_json(root / "manifest.json")
    expected_layers = list(range(num_layers))
    checks = {
        "format": "mage-qwen-prefill-stack-v1",
        "seq_len": sequence,
        "layers": expected_layers,
        "attention_mode": "arm-causal-hxs",
        "activation_granularity": "scalar",
        "outlier_top_k": top_k,
        "token_group_boundaries": list(boundaries),
        "token_hybrid_qkv_start_layer": qkv_start,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{root}/manifest.json: {key}={manifest.get(key)!r}, "
                f"expected {expected!r}"
            )
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict) or sequence not in calibration.get(
        "valid_seq_lens", []
    ):
        raise RuntimeError(f"{root}: complete flexible calibration proof is absent")


def validate_decoder_output(root: Path, num_layers: int) -> None:
    for name in ("decoder.w8a8.onnx", "final_head.w8a8.onnx", "manifest.json"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"decoder export is incomplete: {path}")
    shards = sorted((root / "layers").glob("*.data"))
    if len(shards) != num_layers + 1:
        raise RuntimeError(
            f"decoder has {len(shards)} external-data shards, expected "
            f"{num_layers + 1}"
        )
    manifest = load_json(root / "manifest.json")
    if manifest.get("format") != "mage-qwen3-ort-decoder-v1":
        raise RuntimeError("decoder manifest format is invalid")
    if int(manifest.get("config", {}).get("num_hidden_layers", -1)) != num_layers:
        raise RuntimeError("decoder manifest layer count is invalid")
    if manifest.get("final_head_model") != "final_head.w8a8.onnx":
        raise RuntimeError("decoder manifest does not expose the W8A8 final head")


def publish_directory(stage: Path, destination: Path, force: bool) -> None:
    if not stage.is_dir():
        raise RuntimeError(f"staged directory does not exist: {stage}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.with_name(f".{destination.name}.export-incoming")
    backup = destination.with_name(f".{destination.name}.export-backup")
    if incoming.exists() or backup.exists():
        raise RuntimeError(
            f"stale publish directory exists: {incoming} or {backup}; "
            "inspect it before retrying"
        )
    if destination.exists() and not force:
        raise FileExistsError(
            f"refusing to replace {destination}; pass --force after checking "
            "the staged export"
        )
    if stage.stat().st_dev == destination.parent.stat().st_dev:
        stage.rename(incoming)
    else:
        shutil.copytree(stage, incoming)
    replaced = False
    try:
        if destination.exists():
            destination.rename(backup)
            replaced = True
        incoming.rename(destination)
    except BaseException:
        if replaced and not destination.exists() and backup.exists():
            backup.rename(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
        print(f"published {destination}")


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MAGE_ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"deployment asset is outside {MAGE_ROOT}: {path}") from error


def refresh_assets(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("assets.json can only describe deploy/models")
    vision = output_root / "vision" / VISION_NAME
    decoder = output_root / "qwen-decode-ort" / "w8a8"
    decoder_manifest = load_json(decoder / "manifest.json")
    num_layers = int(decoder_manifest.get("config", {}).get("num_hidden_layers", -1))
    if num_layers <= 0:
        raise RuntimeError("cannot infer decoder layer count")
    profiles: dict[str, dict[str, Any]] = {}
    hash_paths = [
        vision,
        decoder / "decoder.w8a8.onnx",
        decoder / "final_head.w8a8.onnx",
        decoder / "manifest.json",
        DEFAULT_ADDON,
    ]
    for top_k in PREFILL_TOPKS:
        name = profile_name(1024, top_k, 12)
        root = output_root / "qwen-prefill" / name
        manifest_path = root / "manifest.json"
        validate_prefill_output(root, 1024, num_layers, top_k, (21,), 12)
        profiles[name] = {
            "directory": _relative_to_project(root),
            "seq_len": 1024,
            "expected_fb_count": num_layers,
        }
        hash_paths.append(manifest_path)
    # Older deployment packages predate the automatic vision audit. New
    # exports always contain it, but refreshing hashes must remain compatible
    # with an otherwise valid already-deployed package.
    validate_vision_output(output_root / "vision", require_audit=False)
    validate_decoder_output(decoder, num_layers)
    for path in hash_paths:
        if not path.is_file():
            raise RuntimeError(f"cannot hash missing deployment asset: {path}")
    primary = profile_name(1024, 4, 12)
    manifest = {
        "format": "mage-vl-deployment-assets-v1",
        "profile": primary,
        "expected_prefill_fb_count": num_layers,
        "expected_decoder_data_count": num_layers + 1,
        "prefill_profiles": profiles,
        "checkpoint_fingerprint": decoder_manifest.get("checkpoint_fingerprint"),
        "sha256": {
            _relative_to_project(path): sha256(path) for path in hash_paths
        },
    }
    write_json_atomic(DEPLOY_DIR / "assets.json", manifest)
    return manifest


def build_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).resolve()
    checkpoint = validate_checkpoint(model_path)
    config = checkpoint["config"]
    text_config = config["text_config"]
    num_layers = int(text_config["num_hidden_layers"])
    hidden_size = int(text_config["hidden_size"])
    image_token_id = int(config["image_token_id"])
    components = tuple(dict.fromkeys(args.component or ("vision", "prefill", "decoder")))
    topks = tuple(dict.fromkeys(args.prefill_top_k or PREFILL_TOPKS))
    boundaries = tuple(args.token_group_boundaries)
    if not boundaries or list(boundaries) != sorted(set(boundaries)):
        raise ValueError("--token-group-boundaries must be sorted and unique")
    if args.seq_len != 1024:
        raise ValueError("this deployment package currently supports only S=1024")
    addon = Path(args.addon_path).resolve()
    if not addon.is_file():
        raise ValueError(f"addon does not exist: {addon}")
    output_root = Path(args.output_root).resolve()
    work = Path(args.work_dir).resolve()
    stage_root = work / "staging-models"
    log_path = work / "export.log"
    if args.vision_outlier_top_k != 2 and "vision" in components:
        raise ValueError(
            "the deployment filename/config currently requires visual Top-K=2"
        )
    if (
        output_root == DEFAULT_OUTPUT_ROOT.resolve()
        and not args.no_update_assets
        and (
            boundaries != (21,)
            or args.token_hybrid_qkv_start_layer != 12
        )
    ):
        raise ValueError(
            "deploy/assets.json describes the [21], QKV@12 profiles; use the "
            "stock topology, or pass --no-update-assets for an experimental "
            "profile"
        )

    vision_bundles: list[Path] = []
    prompts: list[Path] = []
    valid_lengths: list[int] = []
    systems: list[str] = []
    if "vision" in components:
        vision_bundles = validate_vision_bundles(
            args.vision_calibration_bundle or (), args.max_vision_calib
        )
    if "prefill" in components:
        prompts, valid_lengths, systems = validate_prompt_dirs(
            args.prefill_prompt_dir or (),
            args.seq_len,
            hidden_size,
            image_token_id,
            boundaries,
        )

    destinations: list[tuple[Path, Path]] = []
    if "vision" in components:
        destinations.append((stage_root / "vision", output_root / "vision"))
    if "prefill" in components:
        for top_k in topks:
            name = profile_name(args.seq_len, top_k, args.token_hybrid_qkv_start_layer)
            destinations.append(
                (
                    stage_root / "qwen-prefill" / name,
                    output_root / "qwen-prefill" / name,
                )
            )
    if "decoder" in components:
        destinations.append(
            (
                stage_root / "qwen-decode-ort" / "w8a8",
                output_root / "qwen-decode-ort" / "w8a8",
            )
        )
    if not args.dry_run:
        existing = [destination for _, destination in destinations if destination.exists()]
        if existing and not args.force:
            raise FileExistsError(
                "the following deployed directories already exist; rerun with "
                "--force to replace them only after the complete staged build "
                "passes:\n  - " + "\n  - ".join(map(str, existing))
            )
        if stage_root.exists():
            if not args.force:
                raise FileExistsError(
                    f"staging directory exists: {stage_root}; inspect it or rerun "
                    "with --force"
                )
            shutil.rmtree(stage_root)
        stage_root.mkdir(parents=True)
        if args.force and log_path.exists():
            log_path.unlink()

    runner = CommandRunner(args.dry_run, log_path)
    # Keep a virtualenv interpreter path intact. Path.resolve() follows its
    # symlink to /usr/bin/python and would silently discard the venv packages.
    python = str(Path(args.python).absolute()) if os.sep in args.python else args.python
    started = time.time()

    if "vision" in components:
        vision_root = stage_root / "vision"
        if not args.dry_run:
            vision_root.mkdir(parents=True, exist_ok=True)
        command = [
            python,
            str(MAGE_ROOT / "python" / "build_mage_vit.py"),
            "--model-path",
            str(model_path),
            "--canvas-size",
            "288",
            "512",
        ]
        for bundle in vision_bundles:
            command.extend(("--bundle", str(bundle)))
        command.extend(
            (
                "--max-calib",
                str(args.max_vision_calib),
                "--device",
                args.device,
                "--dump-ranges",
                str(work / "vision.ranges.json"),
                "--dump-quant-fb",
                str(vision_root / VISION_NAME),
                "--quant-profile",
                "int8-fp16-topk",
                "--outlier-top-k",
                str(args.vision_outlier_top_k),
                "--per-channel-qk",
                "--per-channel-qk-max-requant-multiplier",
                str(args.max_requant_multiplier),
                "--dump-bypass-report",
                str(vision_root / "build.report.json"),
                "--dump-symbol-map",
                str(vision_root / "symbols.json"),
                "--addon-path",
                str(addon),
            )
        )
        runner.run(command)
        if not args.dry_run:
            audit_vision_fb(
                vision_root / VISION_NAME,
                addon,
                vision_root / "tensor-audit.json",
            )
            rewrite_staged_paths(vision_root, output_root / "vision")
            validate_vision_output(vision_root)

    if "prefill" in components:
        range_json = work / "qwen-prefill.ranges.json"
        calibration_report = work / "qwen-prefill.calibration.json"
        command = [
            python,
            str(MAGE_ROOT / "Qwen-prefill" / "collect_qwen_prefill_ranges.py"),
            "--model-path",
            str(model_path),
        ]
        for prompt in prompts:
            command.extend(("--prompt-dir", str(prompt)))
        command.extend(
            (
                "--device",
                args.device,
                "--dtype",
                args.calibration_dtype,
                "--dump-ranges",
                str(range_json),
                "--output-json",
                str(calibration_report),
            )
        )
        runner.run(command)
        for top_k in topks:
            name = profile_name(args.seq_len, top_k, args.token_hybrid_qkv_start_layer)
            profile_root = stage_root / "qwen-prefill" / name
            command = [
                python,
                str(MAGE_ROOT / "Qwen-prefill" / "build_qwen_prefill_stack.py"),
                "--model-path",
                str(model_path),
                "--seq-len",
                str(args.seq_len),
                "--range-json",
                str(range_json),
                "--calibration-report",
                str(calibration_report),
                "--output-dir",
                str(profile_root),
                "--outlier-top-k",
                str(top_k),
                "--attention-mode",
                "arm-causal-hxs",
                "--activation-granularity",
                "scalar",
                "--per-channel-qk-max-requant-multiplier",
                str(args.max_requant_multiplier),
                "--softmax-threads",
                str(args.softmax_threads),
                "--token-group-boundaries",
                *map(str, boundaries),
                "--token-hybrid-qkv-start-layer",
                str(args.token_hybrid_qkv_start_layer),
                "--addon-path",
                str(addon),
            ]
            for language in args.calibration_language:
                command.extend(("--calibration-language", language))
            runner.run(command)
            destination = output_root / "qwen-prefill" / name
            if not args.dry_run:
                rewrite_staged_paths(profile_root, destination)
                validate_prefill_output(
                    profile_root,
                    args.seq_len,
                    num_layers,
                    top_k,
                    boundaries,
                    args.token_hybrid_qkv_start_layer,
                )

    if "decoder" in components:
        decoder_root = stage_root / "qwen-decode-ort" / "w8a8"
        runner.run(
            [
                python,
                str(MAGE_ROOT / "Qwen-decode-ort" / "build_ort_qwen_decoder.py"),
                "--model-path",
                str(model_path),
                "--output-dir",
                str(decoder_root),
            ]
        )
        if not args.dry_run:
            validate_decoder_output(decoder_root, num_layers)

    report = {
        "format": "mage-vl-model-export-v1",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "model_path": str(model_path),
        "checkpoint_fingerprint": checkpoint["fingerprint"],
        "components": list(components),
        "output_root": str(output_root),
        "work_dir": str(work),
        "vision_calibration_bundles": list(map(str, vision_bundles)),
        "prefill_prompt_dirs": list(map(str, prompts)),
        "prefill_valid_seq_lens": valid_lengths,
        "prefill_system_prompts": list(dict.fromkeys(systems)),
        "prefill_top_k": list(topks),
        "token_group_boundaries": list(boundaries),
        "token_hybrid_qkv_start_layer": args.token_hybrid_qkv_start_layer,
        "commands": [shlex.join(command) for command in runner.commands],
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print("\n" + json.dumps(report, indent=2))
        return

    write_json_atomic(work / "export.report.json", report)
    for stage, destination in destinations:
        publish_directory(stage, destination, args.force)
    write_json_atomic(output_root / "export.report.json", report)
    if (
        output_root == DEFAULT_OUTPUT_ROOT.resolve()
        and not args.no_update_assets
    ):
        refresh_assets(output_root)
        print(f"updated {DEPLOY_DIR / 'assets.json'}")
    elif not args.no_update_assets:
        print("custom --output-root: skipped deploy/assets.json refresh")
    print(json.dumps(report, indent=2))


def _prompt_token_count(tokenizer: Any, vision_content: str, system: str, question: str) -> int:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": vision_content + "\n" + question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Match prepare_qwen_prefill_prompt.py exactly; some custom tokenizers do
    # add their own special token when this argument is left at its default.
    encoded = tokenizer(text)
    return len(encoded["input_ids"])


def make_full_prompt_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).resolve()
    validate_checkpoint(model_path)
    bundle = Path(args.bundle).resolve()
    manifest = load_json(bundle / "manifest.json")
    if not (bundle / "visual_embeddings.f32").is_file():
        raise ValueError(f"{bundle} has no visual_embeddings.f32")
    sys.path.insert(0, str(MAGE_ROOT / "python"))
    import qwen3_bridge
    from transformers import AutoTokenizer

    vision_content, _ = qwen3_bridge.build_vision_content(manifest, args.fps)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    target = args.seq_len

    def question(repetitions: int) -> str:
        tail = " ".join([args.filler] * repetitions)
        return (args.base_question.strip() + " " + tail).strip()

    low = 0
    high = 1
    while _prompt_token_count(
        tokenizer, vision_content, args.system, question(high)
    ) < target:
        high *= 2
        if high > target * 8:
            raise RuntimeError("could not grow the calibration question to the target")
    while low < high:
        middle = (low + high) // 2
        length = _prompt_token_count(
            tokenizer, vision_content, args.system, question(middle)
        )
        if length < target:
            low = middle + 1
        else:
            high = middle
    found = None
    for repetitions in range(max(0, low - 8), low + 9):
        candidate = question(repetitions)
        if _prompt_token_count(tokenizer, vision_content, args.system, candidate) == target:
            found = (repetitions, candidate)
            break
    if found is None:
        raise RuntimeError(
            f"filler {args.filler!r} cannot produce exactly S={target}; "
            "try --filler word or another one-token string"
        )
    repetitions, candidate = found
    output = Path(args.output_dir).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"prompt output exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"refusing to overwrite non-empty prompt directory {output}; "
                "pass --force to replace it"
            )
        forbidden = {
            Path(output.anchor),
            SDK_ROOT.resolve(),
            MAGE_ROOT.resolve(),
            DEPLOY_DIR.resolve(),
        }
        if output in forbidden:
            raise ValueError(f"refusing to remove broad directory: {output}")
        shutil.rmtree(output)
    command = [
        args.python,
        str(MAGE_ROOT / "Qwen-prefill" / "prepare_qwen_prefill_prompt.py"),
        "--model-path",
        str(model_path),
        "--bundle",
        str(bundle),
        "--output-dir",
        str(output),
        "--question",
        candidate,
        "--system",
        args.system,
        "--fps",
        str(args.fps),
        "--pad-to-seq-len",
        str(target),
    ]
    print(
        f"found full-length prompt: filler={args.filler!r} "
        f"repetitions={repetitions} characters={len(candidate)}"
    )
    subprocess.run(command, cwd=SDK_ROOT, check=True)
    metadata = load_json(output / "metadata.json")
    if int(metadata.get("valid_seq_len", -1)) != target:
        raise RuntimeError("prepared prompt is not exactly full length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="build selected components and publish them after validation"
    )
    build.add_argument("--model-path", required=True)
    build.add_argument(
        "--component",
        action="append",
        choices=("vision", "prefill", "decoder"),
        help="component to export; repeat as needed (default: all)",
    )
    build.add_argument("--vision-calibration-bundle", action="append")
    build.add_argument("--prefill-prompt-dir", action="append")
    build.add_argument("--seq-len", type=int, default=1024)
    build.add_argument("--vision-outlier-top-k", type=int, default=2)
    build.add_argument(
        "--prefill-top-k",
        type=int,
        action="append",
        choices=range(0, 37),
        help="prefill Top-K profile; repeat (default: 4 and 0)",
    )
    build.add_argument("--token-group-boundaries", type=int, nargs="+", default=[21])
    build.add_argument("--token-hybrid-qkv-start-layer", type=int, default=12)
    build.add_argument("--max-requant-multiplier", type=float, default=0.99)
    build.add_argument("--softmax-threads", type=int, default=0)
    build.add_argument("--max-vision-calib", type=int, default=0)
    build.add_argument("--calibration-language", action="append", default=[])
    build.add_argument("--device", default="cuda")
    build.add_argument(
        "--calibration-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    build.add_argument("--addon-path", default=str(DEFAULT_ADDON))
    build.add_argument("--python", default=sys.executable)
    build.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    build.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    build.add_argument("--force", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--no-update-assets", action="store_true")
    build.set_defaults(handler=build_command)

    full = subparsers.add_parser(
        "make-full-prompt",
        help="prepare one real S=1024 calibration prompt from an embedding bundle",
    )
    full.add_argument("--model-path", required=True)
    full.add_argument("--bundle", required=True)
    full.add_argument("--output-dir", required=True)
    full.add_argument("--seq-len", type=int, default=1024)
    full.add_argument("--system", default="You are a helpful assistant.")
    full.add_argument("--base-question", default="Describe this video in detail.")
    full.add_argument("--filler", default="word")
    full.add_argument("--fps", type=float, default=24.0)
    full.add_argument("--python", default=sys.executable)
    full.add_argument("--force", action="store_true")
    full.set_defaults(handler=make_full_prompt_command)

    refresh = subparsers.add_parser(
        "refresh-assets", help="validate deploy/models and rewrite assets.json hashes"
    )
    refresh.set_defaults(
        handler=lambda _args: print(json.dumps(refresh_assets(), indent=2))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
