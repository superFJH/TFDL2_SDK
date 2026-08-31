#!/usr/bin/env python3
"""Export the non-layer checkpoint data needed by checkpoint-free inference.

The vision/prefill FBs and ONNX decoder already contain all transformer
weights.  Runtime inference still needs tokenizer/processor files and random
access to the tied token embedding table.  This exporter copies only those
assets and preserves the embedding table's BF16 payload bit-for-bit, without
copying a safetensors shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from runtime_assets import RUNTIME_ASSETS_FORMAT


STATIC_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
EMBEDDING_CANDIDATES = (
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
FINAL_NORM_CANDIDATES = (
    "model.language_model.norm.weight",
    "language_model.model.norm.weight",
    "model.norm.weight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    value = json.loads(index_path.read_text())
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has no weight_map")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def _tensor_metadata(
    model_path: Path, candidates: tuple[str, ...], description: str
) -> tuple[str, Path, dict[str, object]]:
    weight_map = _weight_map(model_path)
    name = next((item for item in candidates if item in weight_map), None)
    if name is None:
        raise KeyError(f"checkpoint has no supported {description} tensor")
    shard = model_path / weight_map[name]
    with shard.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_len))
    metadata = header.get(name)
    if not isinstance(metadata, dict):
        raise ValueError(f"safetensors shard has no metadata for {name}")
    if metadata.get("dtype") != "BF16":
        raise ValueError(f"expected BF16 {description}, got {metadata.get('dtype')}")
    return name, shard, metadata


def _copy_payload(source: Path, metadata: dict[str, object], output: Path) -> int:
    offsets = metadata.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("embedding metadata has invalid data_offsets")
    start, end = (int(value) for value in offsets)
    if end <= start:
        raise ValueError("embedding payload is empty")
    with source.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        stream.seek(8 + header_len + start)
        remaining = end - start
        with output.open("wb") as target:
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise IOError("truncated safetensors embedding payload")
                target.write(block)
                remaining -= len(block)
    return end - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Path(args.model_path).resolve()
    output = Path(args.output_dir).resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"invalid checkpoint directory: {model}")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FILES:
        source = model / name
        if source.is_file():
            shutil.copy2(source, output / name)
    required_static = ("config.json", "tokenizer.json", "preprocessor_config.json")
    missing = [name for name in required_static if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("missing required runtime files: " + ", ".join(missing))

    tensor, shard, metadata = _tensor_metadata(
        model, EMBEDDING_CANDIDATES, "token embedding"
    )
    shape = [int(value) for value in metadata.get("shape", [])]
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"invalid token embedding shape: {shape}")
    embedding = output / "token_embedding.bf16.bin"
    bytes_written = _copy_payload(shard, metadata, embedding)
    expected = shape[0] * shape[1] * 2
    if bytes_written != expected:
        raise ValueError(
            f"BF16 embedding is {bytes_written} bytes, expected {expected} bytes"
        )
    norm_tensor, norm_shard, norm_metadata = _tensor_metadata(
        model, FINAL_NORM_CANDIDATES, "final RMSNorm weight"
    )
    norm_shape = [int(value) for value in norm_metadata.get("shape", [])]
    if norm_shape != [shape[1]]:
        raise ValueError(
            f"final RMSNorm shape {norm_shape} does not match hidden size {shape[1]}"
        )
    final_norm = output / "final_norm.bf16.bin"
    norm_bytes = _copy_payload(norm_shard, norm_metadata, final_norm)
    if norm_bytes != shape[1] * 2:
        raise ValueError("final RMSNorm byte count is invalid")
    manifest = {
        "format": RUNTIME_ASSETS_FORMAT,
        "source_checkpoint": str(model),
        "static_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output.iterdir())
            if path.is_file()
            and path.name not in (embedding.name, final_norm.name, "manifest.json")
        },
        "token_embedding": {
            "file": embedding.name,
            "dtype": "BF16",
            "shape": shape,
            "bytes": bytes_written,
            "sha256": _sha256(embedding),
            "source_tensor": tensor,
            "source_shard": shard.name,
        },
        "final_norm": {
            "file": final_norm.name,
            "dtype": "BF16",
            "shape": norm_shape,
            "bytes": norm_bytes,
            "sha256": _sha256(final_norm),
            "source_tensor": norm_tensor,
            "source_shard": norm_shard.name,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
