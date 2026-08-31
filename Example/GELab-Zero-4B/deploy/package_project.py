#!/usr/bin/env python3
"""Create a relocatable GELab project, optionally including the checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
SDK_ROOT = PROJECT_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--include-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _ignore_factory(*, include_checkpoint: bool):
    """Return a copytree filter for exactly the selected deployment assets.

    ``deploy/model/checkpoint`` is deliberately excluded from the initial
    project-tree copy.  This makes ``--include-checkpoint`` meaningful: an
    original safetensors checkpoint is copied only from the explicit
    ``--model-path`` supplied by the caller, never accidentally because a
    local development tree happens to contain a checkpoint symlink or copy.
    """
    config = json.loads((DEPLOY_DIR / "deployment.json").read_text())
    selected_vision = Path(str(config["tfdl"]["vision"]["fb_dir"])).name
    selected_prefill = Path(str(config["tfdl"]["prefill"]["fb_dir"])).name
    selected_decode = Path(str(config["decode"]["decoder_dir"])).name
    model_parent = DEPLOY_DIR / "model"
    vision_parent = model_parent / "vision"
    prefill_parent = model_parent / "prefill"
    decode_parent = model_parent / "decode"

    def _ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {
            name for name in names if name == "__pycache__" or name.endswith(".pyc")
        }
        current = Path(directory)
        if current == PROJECT_ROOT:
            # Calibration corpora and copied vendor trees are development
            # inputs, not runtime dependencies of the portable deployment.
            ignored.update(("vendor", "Qwen-decode-npu", "calibration"))
        if current == DEPLOY_DIR:
            ignored.add("var")
        if current == model_parent:
            # Do not let a pre-existing local checkpoint bypass the explicit
            # --include-checkpoint gate below.
            ignored.add("checkpoint")
        if current == vision_parent:
            ignored.update(name for name in names if name != selected_vision)
        if current == prefill_parent:
            ignored.update(name for name in names if name != selected_prefill)
        if current == decode_parent:
            ignored.update(name for name in names if name != selected_decode)
        return ignored

    return _ignore


def main() -> None:
    args = parse_args()
    if args.model_path and not args.include_checkpoint:
        raise ValueError(
            "--model-path is only valid with --include-checkpoint; the default "
            "API package uses deploy/model/runtime and needs no checkpoint path"
        )
    config = json.loads((DEPLOY_DIR / "deployment.json").read_text())
    runtime_assets = DEPLOY_DIR / str(
        config.get("runtime_assets_dir", "model/runtime")
    )
    if not (runtime_assets / "manifest.json").is_file():
        raise FileNotFoundError(
            "checkpoint-free runtime assets are missing; run "
            "export_models.py --model-path MODEL_DIR --stage runtime first"
        )
    output = Path(args.output).resolve()
    if output == PROJECT_ROOT or PROJECT_ROOT in output.parents:
        raise ValueError("package output must be outside the source project")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"package output already exists: {output}")
        shutil.rmtree(output)
    target = output / "GELab-Zero-4B"
    shutil.copytree(
        PROJECT_ROOT,
        target,
        ignore=_ignore_factory(include_checkpoint=args.include_checkpoint),
    )

    addon_source = SDK_ROOT / "AddonOps/build/libTFDLAddOn.so"
    if not addon_source.is_file():
        raise FileNotFoundError(f"custom-op library is missing: {addon_source}")
    runtime = target / "deploy/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copy2(addon_source, runtime / addon_source.name)

    decode = config["decode"]
    if decode.get("engine") == "llama.cpp-external-kv":
        wheels = sorted((DEPLOY_DIR / "wheels").glob("tfdl_llmdecode-*.whl"))
        if len(wheels) != 1:
            raise FileNotFoundError(
                "place exactly one target-native tfdl_llmdecode-*.whl in "
                "GELab-Zero-4B/deploy/wheels before packaging"
            )

    if args.include_checkpoint:
        if not args.model_path:
            raise ValueError("--include-checkpoint requires --model-path")
        checkpoint = Path(args.model_path).resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"invalid checkpoint directory: {checkpoint}")
        shutil.copytree(checkpoint, target / "deploy/model/checkpoint", dirs_exist_ok=True)
    manifest = {
        "format": "gelab-zero-4b-portable-project-v1",
        "project": str(target),
        "checkpoint_included": bool(args.include_checkpoint),
        "runtime_assets_included": True,
        "entrypoint": "deploy/run.sh",
        "verification": "python deploy/verify_assets.py --allow-missing-model",
        "requires": [
            "matching TFDL2 userspace/runtime and NPU driver",
            "Python environment containing numpy, transformers and TFDL2",
            "install deploy/wheels/tfdl_llmdecode-*.whl with pip --no-deps",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "package.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
