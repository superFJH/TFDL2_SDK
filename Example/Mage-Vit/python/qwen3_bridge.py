#!/usr/bin/env python3
"""Reference CPU/GPU Qwen3 backend for precomputed TFDL Mage-ViT embeddings.

This loader instantiates only Qwen3 and copies ``model.language_model.*`` plus
``lm_head.weight`` from the Mage-VL checkpoint. The ~330M-parameter vision tower
is therefore not duplicated on the GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


@contextmanager
def default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def mage_key_for_qwen_key(key: str) -> str:
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    if key == "lm_head.weight":
        return key
    raise KeyError(f"unexpected Qwen3 state key: {key}")


def load_qwen3_only(
    model_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
):
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root = Path(model_path)
    full_config = json.loads((root / "config.json").read_text())
    text_values = dict(full_config["text_config"])
    # Transformers 5.x serializes this under rope_parameters; 4.57's Qwen3
    # class still consumes rope_theta directly.
    rope_parameters = text_values.get("rope_parameters") or {}
    if "rope_theta" not in text_values and "rope_theta" in rope_parameters:
        text_values["rope_theta"] = rope_parameters["rope_theta"]
    text_config = Qwen3Config(**text_values)

    # Meta construction plus to_empty avoids an FP32 allocation peak before
    # loading BF16/FP16 weights.
    with default_dtype(dtype), torch.device("meta"):
        model = Qwen3ForCausalLM(text_config)
    model.to_empty(device="cpu")
    # Qwen RoPE inverse frequencies are non-persistent buffers, so they are
    # absent from safetensors. Meta -> to_empty leaves them uninitialized even
    # though every checkpoint parameter is copied correctly. This is mostly
    # invisible on very short text but destroys a 4K+ visual-token prompt.
    # Recreate the buffers from config before loading parameters.
    rope_theta = float(text_config.rope_parameters["rope_theta"])
    rope_dim = int(
        getattr(text_config, "head_dim", None)
        or text_config.hidden_size // text_config.num_attention_heads
    )
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    for module in model.modules():
        if module.__class__.__name__ == "Qwen3RotaryEmbedding":
            module.inv_freq = inv_freq.clone()
            module.original_inv_freq = inv_freq.clone()
            module.attention_scaling = 1.0
    model.to(dtype=dtype)
    destination = model.state_dict()

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = root / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError("no model.safetensors or index found")
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            weight_map = {key: single.name for key in handle.keys()}

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for target_key in destination:
        source_key = mage_key_for_qwen_key(target_key)
        if source_key not in weight_map:
            raise KeyError(f"Mage checkpoint is missing {source_key}")
        by_shard.setdefault(weight_map[source_key], []).append(
            (target_key, source_key)
        )

    with torch.no_grad():
        for shard, pairs in sorted(by_shard.items()):
            print(f"[load] {shard}: {len(pairs)} Qwen tensors")
            with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
                for target_key, source_key in pairs:
                    source = handle.get_tensor(source_key)
                    if source.shape != destination[target_key].shape:
                        raise ValueError(
                            f"shape mismatch for {source_key}: {source.shape} vs "
                            f"{destination[target_key].shape}"
                        )
                    destination[target_key].copy_(source.to(dtype=dtype))
    model.to(device)
    return model.eval(), full_config


def build_vision_content(
    manifest: dict, fallback_fps: float = 24.0
) -> tuple[str, int]:
    if fallback_fps <= 0:
        raise ValueError("fallback_fps must be positive")
    merge = int(manifest.get("spatial_merge_size", 2))
    merge_factor = merge * merge
    spans: list[list[float | int]] = []
    for canvas in manifest["canvases"]:
        positions = canvas["patch_positions"]
        timestamps = canvas.get("patch_timestamps")
        if len(positions) % merge_factor:
            raise ValueError("manifest patch positions are not merge-aligned")
        if timestamps is not None and len(timestamps) != len(positions):
            raise ValueError("manifest patch timestamps are not position-aligned")
        for patch in range(0, len(positions), merge_factor):
            frame_id = int(positions[patch][0])
            timestamp = (
                float(timestamps[patch])
                if timestamps is not None
                else float(frame_id) / fallback_fps
            )
            same = False
            if spans:
                previous_frame = int(spans[-1][0])
                previous_time = float(spans[-1][1])
                same = frame_id == previous_frame and abs(timestamp - previous_time) < 1e-9
            if same:
                spans[-1][2] = int(spans[-1][2]) + 1
            else:
                spans.append([frame_id, timestamp, 1])
    parts: list[str] = []
    total = 0
    for _, timestamp, token_count in spans:
        parts.append(f"<{float(timestamp):.1f} seconds>")
        parts.append("<|vision_start|>")
        parts.append("<|image_pad|>" * int(token_count))
        parts.append("<|vision_end|>\n")
        total += int(token_count)
    return "".join(parts), total


def load_visual_embeddings(
    bundle: str | Path,
    hidden_size: int,
    fallback_fps: float = 24.0,
) -> tuple[torch.Tensor, dict]:
    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text())
    _, expected_tokens = build_vision_content(manifest, fallback_fps)
    raw = np.fromfile(root / "visual_embeddings.f32", dtype=np.float32)
    expected_values = expected_tokens * hidden_size
    if raw.size != expected_values:
        raise ValueError(
            f"visual_embeddings.f32 has {raw.size} values, expected "
            f"{expected_tokens}x{hidden_size}={expected_values}"
        )
    return torch.from_numpy(raw.reshape(expected_tokens, hidden_size)), manifest


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    visual_embeddings: torch.Tensor,
    image_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> tuple[list[int], float, float]:
    embeddings = model.get_input_embeddings()(input_ids)
    image_mask = input_ids.eq(image_token_id)
    if int(image_mask.sum()) != visual_embeddings.shape[0]:
        raise ValueError(
            f"prompt has {int(image_mask.sum())} image tokens but NPU returned "
            f"{visual_embeddings.shape[0]} embeddings"
        )
    embeddings[image_mask] = visual_embeddings.to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    # Match MageVLModel.forward exactly. Relying on Qwen3's implicit
    # cache_position-derived positions is not equivalent for a long
    # inputs_embeds prefill on all supported Transformers releases.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    prefill_begin = time.perf_counter()
    output = model(
        input_ids=None,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        logits_to_keep=1,
    )
    prefill_seconds = time.perf_counter() - prefill_begin
    decode_begin = time.perf_counter()
    next_token = output.logits[:, -1].argmax(dim=-1)
    generated: list[int] = []
    cache = output.past_key_values
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        if token == eos_token_id:
            break
        generated.append(token)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        output = model(
            input_ids=next_token[:, None],
            attention_mask=attention_mask,
            position_ids=(attention_mask.long().sum(dim=-1) - 1)[:, None],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1)
    return generated, prefill_seconds, time.perf_counter() - decode_begin


def parse_dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--question", default="Describe this video.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="Timestamp fallback when the manifest has no patch_timestamps",
    )
    args = parser.parse_args()

    import transformers
    from transformers import AutoTokenizer

    version = tuple(
        int(value) for value in transformers.__version__.split(".")[:2]
    )
    if version < (5, 7):
        raise RuntimeError(
            "Mage-VL requires transformers>=5.7; older Qwen3 cache/position "
            "implementations can produce invalid multimodal generation"
        )

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    model, config = load_qwen3_only(args.model_path, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    visual, manifest = load_visual_embeddings(
        args.bundle, int(config["text_config"]["hidden_size"]), args.fps
    )
    vision_content, _ = build_vision_content(manifest, args.fps)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": vision_content + "\n" + args.question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    generated, prefill_seconds, decode_seconds = greedy_generate(
        model,
        input_ids,
        attention_mask,
        visual.to(device),
        int(config["image_token_id"]),
        int(config["eos_token_id"]),
        args.max_new_tokens,
    )
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(
        json.dumps(
            {
                "prefill_ms": prefill_seconds * 1000.0,
                "decode_ms": decode_seconds * 1000.0,
                "generated_tokens": len(generated),
                "decode_tokens_per_second": (
                    len(generated) / decode_seconds if decode_seconds else 0.0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
