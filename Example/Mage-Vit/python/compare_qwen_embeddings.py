#!/usr/bin/env python3
"""Compare official and TFDL INT8 Mage-ViT embeddings through one Qwen model.

This script requires the model repository's documented Transformers version
(``transformers>=5.7``). It loads the official full Mage-VL class so the
tokenizer, chat template, Qwen implementation and generation cache are shared
by both sides of the A/B test.
"""

from __future__ import annotations

import argparse
import difflib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_QUESTIONS = (
    "Describe this video in detail.",
    "What sport is being played, and what is happening?",
    "What colors are the players' uniforms?",
    "Summarize the sequence of events in chronological order.",
)


def parse_dtype(value: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[value]


def load_manifest(path: str | Path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())


def load_embedding_file(
    path: str | Path, token_count: int, hidden_size: int
) -> torch.Tensor:
    values = np.fromfile(
        Path(path) / "visual_embeddings.f32", dtype=np.float32
    )
    expected = token_count * hidden_size
    if values.size != expected:
        raise ValueError(
            f"{path} has {values.size} embedding values, expected {expected}"
        )
    return torch.from_numpy(values.reshape(token_count, hidden_size))


def all_positions(manifest: dict) -> torch.Tensor:
    return torch.tensor(
        [
            position
            for canvas in manifest["canvases"]
            for position in canvas["patch_positions"]
        ],
        dtype=torch.int64,
    )


def vision_text(positions: torch.Tensor, fps: float) -> tuple[str, int]:
    temporal = positions[:, 0]
    unique, counts = torch.unique_consecutive(temporal, return_counts=True)
    parts: list[str] = []
    total = 0
    for frame, patches in zip(unique.tolist(), counts.tolist()):
        if frame < 0:
            continue
        tokens = int(patches) // 4
        if tokens <= 0:
            continue
        parts.extend(
            (
                f"<{float(frame) / fps:.1f} seconds>",
                "<|vision_start|>",
                "<|image_pad|>" * tokens,
                "<|vision_end|>\n",
            )
        )
        total += tokens
    return "".join(parts), total


def build_prompt(processor, question: str, visual_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    if placeholder not in text:
        raise ValueError("chat template did not emit the Mage video placeholder")
    return text.replace(placeholder, visual_text, 1)


@torch.inference_mode()
def official_visual_embeddings(
    model,
    processor,
    source_bundle: Path,
    manifest: dict,
    positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    max_pixels: int,
) -> tuple[torch.Tensor, dict[str, float | list[int]]]:
    images = [
        Image.open(source_bundle / entry["file"]).convert("RGB")
        for entry in manifest["canvases"]
    ]
    processed = processor.image_processor(
        images=images,
        return_tensors="pt",
        temporal_patch_size=1,
        max_pixels=max_pixels,
    )
    pixels = processed["pixel_values"].to(device=device, dtype=dtype)
    grid = processed["image_grid_thw"].to(device)
    started = time.perf_counter()
    output = model.visual(
        pixels,
        grid_thw=grid,
        patch_positions=positions.to(device),
    ).last_hidden_state
    seconds = time.perf_counter() - started
    output = output.reshape(-1, output.shape[-1]).float().cpu()
    return output, {
        "seconds": seconds,
        "pixel_values_shape": list(pixels.shape),
        "grid_thw_shape": list(grid.shape),
        "output_shape": list(output.shape),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.double().reshape(-1)
    y = b.double().reshape(-1)
    return float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)))


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    visual: torch.Tensor,
    image_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    image_mask = input_ids.eq(image_token_id)
    if int(image_mask.sum()) != visual.shape[0]:
        raise ValueError(
            f"prompt has {int(image_mask.sum())} image tokens, embeddings have "
            f"{visual.shape[0]}"
        )
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings[image_mask] = visual.to(device=device, dtype=dtype)

    prefill_started = time.perf_counter()
    prefill = model(
        input_ids=None,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    initial_logits = prefill.logits[0, -1].float().cpu()

    generate_started = time.perf_counter()
    output = model.generate(
        input_ids=input_ids,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generate_seconds = time.perf_counter() - generate_started
    tokens = output[0, input_ids.shape[1] :].tolist()
    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "tokens": tokens,
        "prefill_seconds": prefill_seconds,
        "generate_seconds": generate_seconds,
        "initial_logits": initial_logits,
    }


def compare_generations(reference: dict, quantized: dict) -> dict:
    ref_tokens = reference["tokens"]
    got_tokens = quantized["tokens"]
    prefix = 0
    for left, right in zip(ref_tokens, got_tokens):
        if left != right:
            break
        prefix += 1
    token_ratio = difflib.SequenceMatcher(
        None, ref_tokens, got_tokens, autojunk=False
    ).ratio()
    text_ratio = difflib.SequenceMatcher(
        None, reference["text"], quantized["text"], autojunk=False
    ).ratio()
    ref_logits = reference.pop("initial_logits")
    got_logits = quantized.pop("initial_logits")
    ref_top10 = set(torch.topk(ref_logits, 10).indices.tolist())
    got_top10 = set(torch.topk(got_logits, 10).indices.tolist())
    return {
        "exact_token_match": ref_tokens == got_tokens,
        "common_prefix_tokens": prefix,
        "token_sequence_ratio": token_ratio,
        "text_sequence_ratio": text_ratio,
        "initial_top1_same": int(ref_logits.argmax()) == int(got_logits.argmax()),
        "initial_top10_overlap": len(ref_top10 & got_top10),
        "initial_logits_cosine": cosine(ref_logits, got_logits),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--int8-bundle", required=True)
    parser.add_argument("--float-bundle", default=None)
    parser.add_argument("--question", action="append", default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max-pixels", type=int, default=150000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor

    major, minor = (int(value) for value in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 7):
        raise RuntimeError("Mage-VL repository requires transformers>=5.7")

    args = build_parser().parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    source = Path(args.source_bundle)
    manifest = load_manifest(source)
    positions = all_positions(manifest)
    visual_prompt, token_count = vision_text(positions, args.fps)

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=str(device),
        attn_implementation="sdpa",
    ).eval()
    official, official_metrics = official_visual_embeddings(
        model,
        processor,
        source,
        manifest,
        positions,
        device,
        dtype,
        args.max_pixels,
    )
    int8 = load_embedding_file(
        args.int8_bundle, token_count, official.shape[-1]
    )
    report: dict[str, object] = {
        "transformers_version": transformers.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "visual_tokens": token_count,
        "official_visual": official_metrics,
        "int8_vs_official_visual_cosine": cosine(official, int8),
        "questions": [],
    }
    if args.float_bundle:
        converted_float = load_embedding_file(
            args.float_bundle, token_count, official.shape[-1]
        )
        report["converted_float_vs_official_visual_cosine"] = cosine(
            official, converted_float
        )

    questions = args.question or list(DEFAULT_QUESTIONS)
    tokenizer = processor.tokenizer
    image_token_id = int(model.config.image_token_id)
    for index, question in enumerate(questions):
        prompt = build_prompt(processor, question, visual_prompt)
        print(f"[qwen] question {index + 1}/{len(questions)}: {question}", flush=True)
        reference = generate_one(
            model, tokenizer, prompt, official, image_token_id,
            device, dtype, args.max_new_tokens,
        )
        quantized = generate_one(
            model, tokenizer, prompt, int8, image_token_id,
            device, dtype, args.max_new_tokens,
        )
        comparison = compare_generations(reference, quantized)
        item = {
            "question": question,
            "official": reference,
            "int8": quantized,
            "comparison": comparison,
        }
        report["questions"].append(item)  # type: ignore[union-attr]
        print("[official]", reference["text"], flush=True)
        print("[int8]", quantized["text"], flush=True)
        print("[compare]", json.dumps(comparison, sort_keys=True), flush=True)

    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
