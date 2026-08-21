#!/usr/bin/env python3
"""Continue greedy CPU/GPU decode from the NPU-prefill FP16 KV bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


MEGAVIT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(MEGAVIT_PYTHON) not in sys.path:
    sys.path.insert(0, str(MEGAVIT_PYTHON))
import qwen3_bridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--prefill-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import transformers
    from transformers import AutoTokenizer, DynamicCache

    version = tuple(
        int(value) for value in transformers.__version__.split(".")[:2]
    )
    if version < (4, 57):
        raise RuntimeError(
            "Mage-VL decode requires transformers>=4.57; use the same "
            "environment as qwen3_bridge.py"
        )
    device = torch.device(args.device)
    dtype = qwen3_bridge.parse_dtype(args.dtype)
    model, raw_config = qwen3_bridge.load_qwen3_only(
        args.model_path, device, dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    prompt = Path(args.prompt_dir)
    prefill_root = Path(args.prefill_dir)
    manifest = json.loads((prefill_root / "manifest.json").read_text())
    input_ids = torch.from_numpy(np.load(prompt / "input_ids.npy")).to(
        device=device, dtype=torch.long
    )
    attention_mask = torch.from_numpy(
        np.load(prompt / "attention_mask.npy")
    ).to(device=device, dtype=torch.long)
    logits = torch.from_numpy(
        np.load(prefill_root / "last_token_logits.npy")
    ).to(device=device)
    cache = DynamicCache()
    expected_layers = list(range(len(manifest["cache_files"])))
    layers = [int(value) for value in manifest["layers"]]
    if layers != expected_layers:
        raise ValueError(
            "decode requires a complete ordered cache for every Qwen layer"
        )
    for layer in layers:
        item = manifest["cache_files"][str(layer)]
        key = torch.from_numpy(np.load(prefill_root / item["key"])).to(
            device=device, dtype=dtype
        )
        value = torch.from_numpy(np.load(prefill_root / item["value"])).to(
            device=device, dtype=dtype
        )
        cache.update(key, value, layer)

    eos_token_id = int(raw_config["eos_token_id"])
    next_token = logits.argmax().reshape(1)
    generated: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            token = int(next_token.item())
            if token == eos_token_id:
                break
            generated.append(token)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            output = model(
                input_ids=next_token.reshape(1, 1),
                attention_mask=attention_mask,
                position_ids=(
                    attention_mask.long().sum(dim=-1) - 1
                )[:, None],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1)
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    report = {
        "model_path": args.model_path,
        "prompt_dir": args.prompt_dir,
        "prefill_dir": args.prefill_dir,
        "device": str(device),
        "dtype": args.dtype,
        "generated_tokens": len(generated),
        "decode_seconds": elapsed,
        "decode_tokens_per_second": len(generated) / elapsed if elapsed else 0.0,
        "tokens": generated,
        "text": text,
    }
    print(text)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"tokens", "text"}}, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
