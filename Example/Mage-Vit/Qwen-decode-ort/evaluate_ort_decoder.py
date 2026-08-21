#!/usr/bin/env python3
"""Compare ORT W8A8 decode with PyTorch from the identical external KV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import decode_ort as runtime
import ort_qwen_decoder as decoder


MEGAVIT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(MEGAVIT_PYTHON) not in sys.path:
    sys.path.insert(0, str(MEGAVIT_PYTHON))
import qwen3_bridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--prefill-dir", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--reference-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(actual, dtype=np.float64).reshape(-1)
    delta = right - left
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "cosine": float(np.dot(left, right) / denominator) if denominator else 0.0,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rel_l2": float(np.linalg.norm(delta) / np.linalg.norm(left)),
    }


def _top(logits: np.ndarray, count: int = 10) -> list[int]:
    return [int(value) for value in np.argsort(logits.reshape(-1))[-count:][::-1]]


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    model_root = Path(args.model_path)
    decoder_root = Path(args.decoder_dir)
    prompt_root = Path(args.prompt_dir)
    prefill_root = Path(args.prefill_dir)
    config = decoder.prefill.QwenPrefillConfig.from_model(model_root)
    decoder_manifest = json.loads((decoder_root / "manifest.json").read_text())
    prefill_manifest = json.loads((prefill_root / "manifest.json").read_text())
    prompt_ids = np.load(prompt_root / "input_ids.npy")
    attention_mask_np = np.load(prompt_root / "attention_mask.npy")
    position_ids = np.load(prompt_root / "position_ids.npy")
    sequence = int(prompt_ids.shape[1])
    cache_capacity = sequence + args.steps
    ort_cache = runtime.ExternalKvCache(
        prefill_root,
        prefill_manifest,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        capacity=cache_capacity,
    )

    session_args = argparse.Namespace(
        threads=args.threads,
        inter_op_threads=1,
        disable_spinning=False,
    )
    session, session_load_seconds = runtime._make_session(
        decoder_root / str(decoder_manifest["model_file"]), session_args
    )
    runtime._validate_session(session, config)
    output_names = [item.name for item in session.get_outputs()]
    embedding = runtime.EmbeddingReader(model_root)

    dtype = qwen3_bridge.parse_dtype(args.reference_dtype)
    model_started = time.perf_counter()
    model, _ = qwen3_bridge.load_qwen3_only(model_root, torch.device("cpu"), dtype)
    model_load_seconds = time.perf_counter() - model_started
    from transformers import DynamicCache

    torch_cache = DynamicCache()
    for layer in range(config.num_hidden_layers):
        item = prefill_manifest["cache_files"][str(layer)]
        key = torch.from_numpy(np.load(prefill_root / item["key"])).to(dtype)
        value = torch.from_numpy(np.load(prefill_root / item["value"])).to(dtype)
        torch_cache.update(key, value, layer)
    attention_mask = torch.from_numpy(attention_mask_np).long()
    seed_logits = np.load(prefill_root / "last_token_logits.npy")
    token = int(seed_logits.reshape(-1).argmax())
    first_position = int(position_ids[0, -1]) + 1
    reports = []

    with torch.inference_mode():
        for step in range(args.steps):
            position = first_position + step
            sin, cos = decoder.prefill.compute_rope(
                np.asarray([position]), config.head_dim, config.rope_theta
            )
            feeds = ort_cache.feeds()
            feeds[decoder.HIDDEN_INPUT] = embedding.get(token)
            feeds[decoder.SIN_INPUT] = sin
            feeds[decoder.COS_INPUT] = cos
            ort_started = time.perf_counter()
            ort_values = session.run(output_names, feeds)
            ort_seconds = time.perf_counter() - ort_started
            ort_outputs = dict(zip(output_names, ort_values))

            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=torch.long)), dim=1
            )
            torch_started = time.perf_counter()
            torch_output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                attention_mask=attention_mask,
                position_ids=torch.tensor([[position]], dtype=torch.long),
                past_key_values=torch_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            torch_seconds = time.perf_counter() - torch_started
            torch_cache = torch_output.past_key_values
            reference_logits = torch_output.logits.float().cpu().numpy()
            actual_logits = ort_outputs[decoder.LOGITS_OUTPUT]
            reference_top10 = _top(reference_logits)
            actual_top10 = _top(actual_logits)
            legacy_cache = torch_cache.to_legacy_cache()
            key_cosines = []
            value_cosines = []
            for layer in range(config.num_hidden_layers):
                reference_key = (
                    legacy_cache[layer][0][:, :, -1:, :]
                    .transpose(1, 2)
                    .float()
                    .cpu()
                    .numpy()
                )
                reference_value = (
                    legacy_cache[layer][1][:, :, -1:, :]
                    .transpose(1, 2)
                    .float()
                    .cpu()
                    .numpy()
                )
                key_cosines.append(
                    metrics(
                        reference_key, ort_outputs[decoder.present_key_name(layer)]
                    )["cosine"]
                )
                value_cosines.append(
                    metrics(
                        reference_value,
                        ort_outputs[decoder.present_value_name(layer)],
                    )["cosine"]
                )
            item = {
                "step": step,
                "input_token": token,
                "position": position,
                "ort_seconds": ort_seconds,
                "torch_seconds": torch_seconds,
                "logits": metrics(reference_logits, actual_logits),
                "reference_top1": reference_top10[0],
                "ort_top1": actual_top10[0],
                "top1_agreement": reference_top10[0] == actual_top10[0],
                "top10_overlap": len(set(reference_top10) & set(actual_top10)),
                "reference_top10": reference_top10,
                "ort_top10": actual_top10,
                "key_cosine_mean": float(np.mean(key_cosines)),
                "key_cosine_min": float(np.min(key_cosines)),
                "value_cosine_mean": float(np.mean(value_cosines)),
                "value_cosine_min": float(np.min(value_cosines)),
            }
            reports.append(item)
            print(json.dumps(item), flush=True)
            ort_cache.append(ort_outputs)
            # Teacher-force the reference Top-1 so both engines always consume
            # the same token and differences remain attributable to execution.
            token = reference_top10[0]

    report = {
        "format": "mage-qwen3-ort-vs-pytorch-v1",
        "model_path": str(model_root),
        "decoder_dir": str(decoder_root),
        "prompt_dir": str(prompt_root),
        "prefill_dir": str(prefill_root),
        "reference_dtype": args.reference_dtype,
        "threads": args.threads,
        "session_load_seconds": session_load_seconds,
        "reference_model_load_seconds": model_load_seconds,
        "steps": reports,
        "summary": {
            "logits_cosine_mean": float(
                np.mean([item["logits"]["cosine"] for item in reports])
            ),
            "top1_agreement": int(sum(item["top1_agreement"] for item in reports)),
            "top10_overlap_mean": float(
                np.mean([item["top10_overlap"] for item in reports])
            ),
            "ort_tokens_per_second": len(reports)
            / sum(item["ort_seconds"] for item in reports),
        },
    }
    path = (
        Path(args.output_json)
        if args.output_json
        else prefill_root / "ort_vs_pytorch.report.json"
    )
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
