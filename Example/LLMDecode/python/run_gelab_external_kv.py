#!/usr/bin/env python3
"""Run llama.cpp decode from a GELab TFDL prefill KV bundle.

The native binary deliberately receives only mmap-able NPY tensors and scalar
geometry.  Keeping the JSON/NumPy manifest parsing here makes the C++ ABI
small, lets it stay useful for any accelerator, and avoids copying the 36
layers of FP16 KV data into another Python cache.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a GELab external TFDL FP16 KV prefix with llama.cpp."
    )
    parser.add_argument("--model", required=True, type=Path, help="exact decoder GGUF")
    parser.add_argument("--prefill-dir", required=True, type=Path)
    parser.add_argument("--prompt-dir", required=True, type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=PROJECT_ROOT / "build" / "bin" / "llmdecode",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--kv-cache-type",
        choices=("q8_0", "fp16"),
        default="fp16",
        help="llama.cpp cache type; external TFDL KV remains FP16 (q8_0 is experimental)",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional native result path; stdout is always retained",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    prefill_dir = args.prefill_dir.resolve()
    prompt_dir = args.prompt_dir.resolve()
    manifest = read_json(prefill_dir / "manifest.json")
    metadata = read_json(prompt_dir / "metadata.json")
    if manifest.get("format") != "mage-qwen-prefill-kv-v1":
        raise ValueError("prefill directory is not a GELab resident TFDL KV bundle")
    if metadata.get("format") != "qwen3-vl-prefill-prompt-v1":
        raise ValueError("prompt directory is not a GELab MRoPE prompt bundle")

    layers = [int(item) for item in manifest["layers"]]
    if layers != list(range(len(layers))):
        raise ValueError("external decode requires a complete ordered KV bundle")
    prompt_tokens = int(manifest["valid_seq_len"])
    if prompt_tokens != int(metadata["valid_seq_len"]):
        raise ValueError("prefill and prompt valid sequence lengths differ")
    kv_heads = int(manifest["num_key_value_heads"])
    head_dim = int(manifest["head_dim"])
    rope_delta = int(metadata["rope_delta"])
    first_decode_position = prompt_tokens + rope_delta

    command = [
        str(args.binary),
        "--model",
        str(args.model.resolve()),
        "--logits",
        str(prefill_dir / str(manifest["last_token_logits"])),
        "--positions",
        str(prompt_dir / str(metadata["files"]["position_ids_3d"])),
        "--layers",
        str(len(layers)),
        "--kv-heads",
        str(kv_heads),
        "--head-dim",
        str(head_dim),
        "--prompt-tokens",
        str(prompt_tokens),
        "--first-decode-position",
        str(first_decode_position),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--kv-cache-type",
        args.kv_cache_type,
    ]
    if args.threads > 0:
        command.extend(["--threads", str(args.threads)])
    cache_files = manifest["cache_files"]
    for layer in layers:
        entry = cache_files[str(layer)]
        key = prefill_dir / str(entry["key"])
        value = prefill_dir / str(entry["value"])
        # Validate before starting a model load.  mmap_mode confirms the ABI
        # while preserving the native binary's zero-copy data source.
        expected = (1, kv_heads, prompt_tokens, head_dim)
        for kind, path in (("key", key), ("value", value)):
            tensor = np.load(path, mmap_mode="r")
            if tensor.dtype != np.float16 or tensor.shape != expected:
                raise ValueError(
                    f"layer {layer} {kind}: expected FP16 {expected}, "
                    f"got {tensor.dtype} {tensor.shape}"
                )
        command.extend(["--key", str(key), "--value", str(value)])

    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            sys.stderr.write(error.stderr)
        raise
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "engine": "llama.cpp-external-kv-fp16",
                    "model": str(args.model.resolve()),
                    "prefill_dir": str(prefill_dir),
                    "prompt_dir": str(prompt_dir),
                    "rope_delta": rope_delta,
                    "kv_cache_type": args.kv_cache_type,
                    "native": payload,
                    "native_stderr": completed.stderr,
                },
                indent=2,
            )
        )
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
