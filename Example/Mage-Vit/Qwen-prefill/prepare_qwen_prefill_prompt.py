#!/usr/bin/env python3
"""Assemble a real Mage-VL prompt into the fixed-prefill tensor ABI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


MEGAVIT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(MEGAVIT_PYTHON) not in sys.path:
    sys.path.insert(0, str(MEGAVIT_PYTHON))
import qwen3_bridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question", default="Describe this video.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--pad-to-seq-len",
        type=int,
        help=(
            "right-pad the assembled prompt to this fixed NPU sequence "
            "bucket; metadata retains the real valid token count"
        ),
    )
    return parser.parse_args()


def right_pad_prompt(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    target_sequence: int | None,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Right-pad one unpadded prompt while retaining its real length."""
    ids = np.asarray(input_ids, dtype=np.int64)
    mask = np.asarray(attention_mask, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != mask.shape:
        raise ValueError(
            "input_ids and attention_mask must have identical [1,S] shapes"
        )
    if not np.all(mask == 1):
        raise ValueError("the tokenizer unexpectedly returned a padded prompt")
    valid_sequence = int(ids.shape[1])
    if target_sequence is None:
        return np.ascontiguousarray(ids), np.ascontiguousarray(mask), valid_sequence
    if target_sequence <= 0:
        raise ValueError("--pad-to-seq-len must be positive")
    if valid_sequence > target_sequence:
        raise ValueError(
            f"prompt needs {valid_sequence} tokens, but the selected NPU "
            f"bucket supports at most {target_sequence}"
        )
    padded_ids = np.full((1, target_sequence), pad_token_id, dtype=np.int64)
    padded_mask = np.zeros((1, target_sequence), dtype=np.int64)
    padded_ids[:, :valid_sequence] = ids
    padded_mask[:, :valid_sequence] = mask
    return padded_ids, padded_mask, valid_sequence


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    root = Path(args.model_path)
    raw_config = json.loads((root / "config.json").read_text())
    config = prefill.QwenPrefillConfig.from_model(root)
    tokenizer = AutoTokenizer.from_pretrained(
        root, trust_remote_code=True
    )
    visual_tensor, manifest = qwen3_bridge.load_visual_embeddings(
        args.bundle, config.hidden_size, args.fps
    )
    visual = np.ascontiguousarray(visual_tensor.numpy(), dtype=np.float32)
    vision_content, visual_tokens = qwen3_bridge.build_vision_content(
        manifest, args.fps
    )
    messages = [
        {"role": "system", "content": args.system},
        {
            "role": "user",
            "content": vision_content + "\n" + args.question,
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, return_tensors="np")
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else int(raw_config["eos_token_id"])
    )
    input_ids, attention_mask, valid_sequence = right_pad_prompt(
        np.asarray(encoded.input_ids, dtype=np.int64),
        np.asarray(encoded.attention_mask, dtype=np.int64),
        args.pad_to_seq_len,
        pad_token_id,
    )
    index = prefill.SafeTensorIndex(root)
    embedding_name = prefill.embedding_weight_name(index)
    unique_ids, inverse = np.unique(input_ids.reshape(-1), return_inverse=True)
    unique_embeddings = index.read_rows(embedding_name, unique_ids)
    hidden = unique_embeddings[inverse].reshape(
        1, input_ids.shape[1], config.hidden_size
    )
    image_token_id = int(raw_config["image_token_id"])
    image_mask = (input_ids == image_token_id) & (attention_mask == 1)
    if int(np.count_nonzero(image_mask)) != visual.shape[0]:
        raise ValueError(
            f"prompt has {int(np.count_nonzero(image_mask))} image tokens, "
            f"bundle has {visual.shape[0]} visual embeddings"
        )
    hidden[image_mask] = visual
    position_ids = attention_mask.cumsum(axis=-1) - 1
    position_ids[attention_mask == 0] = 1

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "hidden.npy", np.ascontiguousarray(hidden, dtype=np.float32))
    np.save(output / "position_ids.npy", position_ids.astype(np.int64))
    np.save(output / "input_ids.npy", input_ids)
    np.save(output / "attention_mask.npy", attention_mask)
    metadata = {
        "model_path": str(root),
        "bundle": str(Path(args.bundle)),
        "question": args.question,
        "system": args.system,
        # seq_len remains the fixed graph width for compatibility with the
        # FB builders. Consumers that handle KV/cache positions use
        # valid_seq_len instead.
        "seq_len": int(input_ids.shape[1]),
        "model_seq_len": int(input_ids.shape[1]),
        "valid_seq_len": valid_sequence,
        "padding_side": "right",
        "pad_token_id": pad_token_id,
        "visual_tokens": int(visual_tokens),
        "hidden_size": config.hidden_size,
        "image_token_id": image_token_id,
        "eos_token_id": int(raw_config["eos_token_id"]),
        "embedding_weight": embedding_name,
        "files": {
            "hidden": "hidden.npy",
            "position_ids": "position_ids.npy",
            "input_ids": "input_ids.npy",
            "attention_mask": "attention_mask.npy",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
