#!/usr/bin/env python3
"""Verify the deployable FB/ONNX/addon bundle after copying it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((DEPLOY_DIR / "assets.json").read_text())
    deployment = json.loads((DEPLOY_DIR / "deployment.json").read_text())
    bucket = deployment["prefill_bucket"]
    sequence = int(bucket["seq_len"])
    prefill_root = PROJECT_ROOT / bucket["fb_dir"]
    errors: list[str] = []
    for relative, expected in manifest["sha256"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"sha256 mismatch: {relative}")
    prefill_count = len(
        list(
            prefill_root.glob(f"layer_*_seq_{sequence}.fb")
        )
    )
    if prefill_count != int(manifest["expected_prefill_fb_count"]):
        errors.append(f"prefill FB count is {prefill_count}, expected 36")
    profile_counts: dict[str, int] = {}
    for name, spec in manifest.get("prefill_profiles", {}).items():
        root = PROJECT_ROOT / spec["directory"]
        profile_sequence = int(spec["seq_len"])
        expected_count = int(spec["expected_fb_count"])
        count = len(list(root.glob(f"layer_*_seq_{profile_sequence}.fb")))
        profile_counts[name] = count
        if count != expected_count:
            errors.append(
                f"prefill profile {name} has {count} FBs, "
                f"expected {expected_count}"
            )
    decoder_count = len(
        list((DEPLOY_DIR / "models/qwen-decode-ort/w8a8/layers").glob("*.data"))
    )
    if decoder_count != int(manifest["expected_decoder_data_count"]):
        errors.append(f"decoder data count is {decoder_count}, expected 37")
    result = {
        "ok": not errors,
        "profile": manifest["profile"],
        "prefill_fb_count": prefill_count,
        "prefill_profiles": profile_counts,
        "decoder_data_count": decoder_count,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
