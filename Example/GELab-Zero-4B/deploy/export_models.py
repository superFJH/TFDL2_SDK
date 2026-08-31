#!/usr/bin/env python3
"""Export FP16 NPU stages plus the selected checkpoint-free CPU decoder."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
SDK_ROOT = PROJECT_ROOT.parents[1]


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _run(command: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        # The parent shell might have only the NPU driver directory in its
        # loader path.  TFDL's Python extension also needs the SDK userspace
        # library directory, so propagate it explicitly to every exporter.
        environment = dict(os.environ)
        sdk_lib = str(SDK_ROOT / "lib")
        inherited = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            sdk_lib if not inherited else sdk_lib + ":" + inherited
        )
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--stage",
        choices=("inspect", "vision", "prefill", "decode", "runtime", "all"),
        default="all",
    )
    parser.add_argument("--config", default=str(DEPLOY_DIR / "deployment.json"))
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _load(config_path)
    root = config_path.parent
    model = Path(args.model_path).resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"invalid checkpoint directory: {model}")
    python = str(Path(sys.executable))
    tfdl = config["tfdl"]
    addon = _resolve(root, str(tfdl["addon"]))
    addon_source = PROJECT_ROOT.parents[1] / "AddonOps/build/libTFDLAddOn.so"
    addon_for_build = addon if addon.is_file() else addon_source
    if not args.dry_run and not addon_for_build.is_file():
        raise FileNotFoundError(f"custom-op library is missing: {addon_for_build}")

    print(
        json.dumps(
            {
                "model": str(model),
                "stage": args.stage,
                "profile": config["profile"],
                "npu_precision": "fp16",
                "decode_profile": config["decode"]["precision"],
            },
            indent=2,
        )
    )
    if args.stage == "inspect":
        _run(
            [python, str(PROJECT_ROOT / "python/inspect_model.py"), "--model-path", str(model)],
            args.dry_run,
        )
        return

    if args.stage in ("vision", "all"):
        vision = tfdl["vision"]
        if args.layer:
            raise ValueError("single-FB vision export does not support --layer")
        if vision["parameterized_linear_lowering"] != "MatMul":
            raise ValueError("the fixed GELab deployment supports only BSC MatMul vision")
        command = [
            python,
            str(PROJECT_ROOT / "python/build_vision_bsc_matmul.py"),
            "--model-path", str(model),
            "--output-dir", str(_resolve(root, str(vision["fb_dir"]))),
            "--grid-h", str(vision["grid_h"]),
            "--grid-w", str(vision["grid_w"]),
            "--addon-path", str(addon_for_build),
        ]
        _run(command, args.dry_run)

    if args.stage in ("prefill", "all"):
        prefill = tfdl["prefill"]
        command = [
            python,
            str(PROJECT_ROOT / "Qwen-prefill/build_prefill.py"),
            "--model-path", str(model),
            "--seq-len", str(prefill["seq_len"]),
            "--output-dir", str(_resolve(root, str(prefill["fb_dir"]))),
            "--addon-path", str(addon_for_build),
        ]
        for layer in args.layer or ():
            command.extend(("--layer", str(layer)))
        _run(command, args.dry_run)

    if args.stage == "decode":
        raise ValueError(
            "the fixed GELab deployment does not re-export decode; retain the "
            "validated deploy/model/decode artifact"
        )

    if args.stage in ("runtime", "all"):
        command = [
            python,
            str(DEPLOY_DIR / "export_runtime_assets.py"),
            "--model-path", str(model),
            "--output-dir", str(
                _resolve(root, str(config.get("runtime_assets_dir", "model/runtime")))
            ),
            "--force",
        ]
        _run(command, args.dry_run)


if __name__ == "__main__":
    main()
