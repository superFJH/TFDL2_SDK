#!/usr/bin/env python3
"""Write the compact GGUF decoder contract used by the portable API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--quantization", default="Q4_K_M")
    args = parser.parse_args()
    model = args.model_path.resolve()
    output = args.output_dir.resolve()
    raw = json.loads((model / "config.json").read_text())
    text = raw.get("text_config", raw)
    model_file = output / args.model_file
    if not model_file.is_file():
        raise FileNotFoundError(f"GGUF output is missing: {model_file}")
    manifest = {
        "format": "tfdl-llmdecode-gguf-v1",
        "engine": "llama.cpp-external-kv",
        "model_file": args.model_file,
        "quantization": args.quantization,
        "cache_dtype": "float16",
        "tie_word_embeddings": bool(text.get("tie_word_embeddings", False)),
        "final_head": "runtime-assets final RMSNorm + tied BF16 token embedding",
        "config": {
            name: int(text[name])
            for name in (
                "hidden_size",
                "num_hidden_layers",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            )
        },
        "external_kv": {
            "source_dtype": "float16",
            "source_layout": "[B,Hkv,S_valid,D]",
            "valid_length": "per-request attention_mask.sum()",
            "padding_policy": "never import padded KV cells",
            "mrope_positions": "[3,1,S_physical], sliced to S_valid by native worker",
        },
    }
    if not manifest["tie_word_embeddings"]:
        raise ValueError("GELab external decoder currently requires tied embeddings")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
