#!/usr/bin/env python3
"""Export a PyTorch reference prefill using the production NPU KV ABI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ort_qwen_decoder as decoder


MEGAVIT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(MEGAVIT_PYTHON) not in sys.path:
    sys.path.insert(0, str(MEGAVIT_PYTHON))
import qwen3_bridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_path)
    prompt_root = Path(args.prompt_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config = decoder.prefill.QwenPrefillConfig.from_model(model_root)
    hidden = np.load(prompt_root / "hidden.npy")
    input_ids = np.load(prompt_root / "input_ids.npy")
    attention_mask = np.load(prompt_root / "attention_mask.npy")
    position_ids = np.load(prompt_root / "position_ids.npy")
    if hidden.shape != (1, input_ids.shape[1], config.hidden_size):
        raise ValueError(f"invalid prompt hidden shape {hidden.shape}")
    if input_ids.shape != attention_mask.shape or input_ids.shape != position_ids.shape:
        raise ValueError("prompt tensors have inconsistent shapes")
    prompt_metadata_path = prompt_root / "metadata.json"
    prompt_metadata = (
        json.loads(prompt_metadata_path.read_text())
        if prompt_metadata_path.exists()
        else {}
    )
    model_sequence = int(input_ids.shape[1])
    valid_sequence = int(
        prompt_metadata.get("valid_seq_len", int(attention_mask.sum()))
    )
    if not 0 < valid_sequence <= model_sequence:
        raise ValueError("prompt has an invalid valid_seq_len")
    expected_mask = np.zeros_like(attention_mask)
    expected_mask[:, :valid_sequence] = 1
    if not np.array_equal(attention_mask, expected_mask):
        raise ValueError("reference prefill requires one right-padded prefix")

    # The fixed NPU graph is padded to its bucket width, but the decoder ABI
    # contains only real prompt tokens.  Running the reference over padding
    # would make logits refer to a PAD token and would export a longer KV
    # cache than the NPU path, invalidating the handoff comparison.
    hidden = np.ascontiguousarray(hidden[:, :valid_sequence])
    attention_mask = np.ascontiguousarray(attention_mask[:, :valid_sequence])
    position_ids = np.ascontiguousarray(position_ids[:, :valid_sequence])

    device = torch.device(args.device)
    dtype = qwen3_bridge.parse_dtype(args.dtype)
    load_started = time.perf_counter()
    model, _ = qwen3_bridge.load_qwen3_only(model_root, device, dtype)
    load_seconds = time.perf_counter() - load_started
    run_started = time.perf_counter()
    with torch.inference_mode():
        result = model(
            inputs_embeds=torch.from_numpy(hidden).to(device=device, dtype=dtype),
            attention_mask=torch.from_numpy(attention_mask).to(
                device=device, dtype=torch.long
            ),
            position_ids=torch.from_numpy(position_ids).to(
                device=device, dtype=torch.long
            ),
            use_cache=True,
            logits_to_keep=1,
        )
    run_seconds = time.perf_counter() - run_started
    cache = result.past_key_values.to_legacy_cache()
    if len(cache) != config.num_hidden_layers:
        raise ValueError(f"reference returned {len(cache)} cache layers")
    cache_files = {}
    for layer, (key, value) in enumerate(cache):
        key_array = key.detach().cpu().to(torch.float16).numpy()
        value_array = value.detach().cpu().to(torch.float16).numpy()
        key_name = f"layer_{layer:02d}.key.npy"
        value_name = f"layer_{layer:02d}.value.npy"
        np.save(output_root / key_name, key_array)
        np.save(output_root / value_name, value_array)
        cache_files[str(layer)] = {"key": key_name, "value": value_name}
    logits = result.logits[:, -1].float().cpu().numpy()
    np.save(output_root / "last_token_logits.npy", logits)
    manifest = {
        "format": decoder.PREFILL_FORMAT,
        "producer": "pytorch-reference",
        "model_path": str(model_root),
        "prompt": prompt_metadata,
        "seq_len": valid_sequence,
        "valid_seq_len": valid_sequence,
        "model_seq_len": model_sequence,
        "layers": list(range(config.num_hidden_layers)),
        "hidden_size": config.hidden_size,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "cache_dtype": "float16",
        "cache_files": cache_files,
        "logits": {
            "top1": int(logits.reshape(-1).argmax()),
            "top10": [
                int(value) for value in np.argsort(logits.reshape(-1))[-10:][::-1]
            ],
        },
        "reference_dtype": args.dtype,
        "reference_device": str(device),
        "model_load_seconds": load_seconds,
        "prefill_seconds": run_seconds,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
