#!/usr/bin/env python3
"""Collect all-layer Qwen prefill ranges from one fixed sequence bucket."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-dir", action="append")
    parser.add_argument("--synthetic-seq-len", type=int)
    parser.add_argument("--synthetic-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--dump-ranges", required=True)
    parser.add_argument("--save-reference-dir")
    parser.add_argument("--output-json")
    return parser.parse_args()


def _load_prompts(
    args: argparse.Namespace, config: prefill.QwenPrefillConfig
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[str]]:
    hidden: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    labels: list[str] = []
    valid_lengths: list[int] = []
    for directory_value in args.prompt_dir or ():
        directory = Path(directory_value)
        value = np.load(directory / "hidden.npy").astype(np.float32)
        position = np.load(directory / "position_ids.npy").reshape(-1)
        hidden.append(value)
        positions.append(position)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        valid_lengths.append(int(metadata.get("valid_seq_len", value.shape[1])))
        labels.append(str(directory))
    if args.synthetic_seq_len is not None:
        if args.synthetic_seq_len <= 0 or args.synthetic_count <= 0:
            raise ValueError("synthetic sequence/count must be positive")
        rng = np.random.default_rng(args.seed)
        for index in range(args.synthetic_count):
            hidden.append(
                rng.normal(
                    scale=args.input_scale,
                    size=(1, args.synthetic_seq_len, config.hidden_size),
                ).astype(np.float32)
            )
            positions.append(np.arange(args.synthetic_seq_len, dtype=np.int64))
            valid_lengths.append(args.synthetic_seq_len)
            labels.append(f"synthetic-{index}")
    if not hidden:
        raise ValueError("pass --prompt-dir and/or --synthetic-seq-len")
    sequence = hidden[0].shape[1]
    for label, value, position, valid in zip(
        labels, hidden, positions, valid_lengths
    ):
        expected = (1, sequence, config.hidden_size)
        if value.shape != expected:
            raise ValueError(f"{label}: hidden {value.shape}, expected {expected}")
        if position.shape != (sequence,):
            raise ValueError(
                f"{label}: positions {position.shape}, expected {(sequence,)}"
            )
        if not 0 < valid <= sequence:
            raise ValueError(
                f"{label}: valid_seq_len={valid}, expected inside [1, {sequence}]"
            )
    return hidden, positions, valid_lengths, labels


def main() -> None:
    args = parse_args()
    import torch

    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    hidden_np, positions, valid_lengths, labels = _load_prompts(args, config)
    sequence = hidden_np[0].shape[1]
    layers = args.layer or list(range(config.num_hidden_layers))
    if len(set(layers)) != len(layers) or any(
        layer < 0 or layer >= config.num_hidden_layers for layer in layers
    ):
        raise ValueError("--layer contains a duplicate or invalid layer")
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)
    hidden = [
        torch.from_numpy(value).to(device=device, dtype=torch_dtype)
        for value in hidden_np
    ]
    rope = [
        tuple(
            torch.from_numpy(value).to(device=device, dtype=torch_dtype)
            for value in prefill.compute_rope(
                position, config.head_dim, config.rope_theta
            )
        )
        for position in positions
    ]
    collector = prefill.RangeCollector()
    checkpoint = prefill.SafeTensorIndex(args.model_path)
    cache_root = (
        Path(args.save_reference_dir)
        if args.save_reference_dir else None
    )
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
        for index in range(len(hidden)):
            prompt_root = cache_root / f"prompt_{index}"
            prompt_root.mkdir(exist_ok=True)
            np.save(prompt_root / "hidden.npy", hidden_np[index])
            np.save(prompt_root / "position_ids.npy", positions[index])
            (prompt_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "label": labels[index],
                        "seq_len": sequence,
                        "valid_seq_len": valid_lengths[index],
                        "hidden_size": config.hidden_size,
                        "synthetic": labels[index].startswith("synthetic-"),
                    },
                    indent=2,
                )
            )
    layer_report: list[dict[str, object]] = []
    begin = time.perf_counter()
    for layer in layers:
        weights = prefill.load_layer_weights(
            args.model_path, layer, checkpoint
        )
        started = time.perf_counter()
        next_hidden = []
        with torch.inference_mode():
            for prompt_index, (value, (sin, cos), valid_sequence) in enumerate(
                zip(hidden, rope, valid_lengths)
            ):
                output, key, cache_value = prefill.torch_layer(
                    config,
                    layer,
                    weights,
                    value,
                    sin,
                    cos,
                    collector,
                    valid_seq_len=valid_sequence,
                )
                next_hidden.append(output)
                if cache_root:
                    prompt_root = cache_root / f"prompt_{prompt_index}"
                    np.save(
                        prompt_root / f"layer_{layer:02d}.hidden.npy",
                        output.detach().to(torch.float16).cpu().numpy(),
                    )
                    np.save(
                        prompt_root / f"layer_{layer:02d}.key.npy",
                        key.detach().to(torch.float16).cpu().numpy(),
                    )
                    np.save(
                        prompt_root / f"layer_{layer:02d}.value.npy",
                        cache_value.detach().to(torch.float16).cpu().numpy(),
                    )
        hidden = next_hidden
        elapsed = time.perf_counter() - started
        layer_report.append({"layer": layer, "seconds": elapsed})
        print(
            f"layer {layer:02d}: {elapsed:.3f}s, "
            f"ranges={len(collector.items)}",
            flush=True,
        )
        del weights
        gc.collect()
    collector.dump(args.dump_ranges)
    if cache_root:
        for prompt_index, value in enumerate(hidden):
            np.save(
                cache_root / f"prompt_{prompt_index}" / "final_hidden.npy",
                value.detach().to(torch.float16).cpu().numpy(),
            )
        manifest = {
            "format": "mage-qwen-prefill-reference-v1",
            "model_seq_len": sequence,
            "valid_seq_lens": valid_lengths,
            "hidden_size": config.hidden_size,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "layers": layers,
            "prompts": labels,
            "dtype": "float16",
        }
        (cache_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
    report = {
        "format": "mage-qwen-prefill-calibration-v1",
        "model_path": args.model_path,
        "seq_len": sequence,
        "prompts": labels,
        "valid_seq_lens": valid_lengths,
        # These are properties of RangeCollector when valid_seq_len is passed.
        "padding_ranges_ignored": True,
        "causal_qk_cells_only": True,
        "layers": layers,
        "device": str(device),
        "dtype": args.dtype,
        "range_count": len(collector.items),
        "row_range_count": len(collector.row_items),
        "token_range_count": len(collector.token_items),
        "dump_ranges": args.dump_ranges,
        "total_seconds": time.perf_counter() - begin,
        "layer_timings": layer_report,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
