#!/usr/bin/env python3
"""Shared Qwen3 one-token decoder primitives for ONNX Runtime.

The exported ABI intentionally stores KV in token-major order
``[batch, past_sequence, kv_heads, head_dim]``.  With batch fixed to one, a
prefix of a preallocated cache remains contiguous, so decode does not copy the
complete cache on every token.  The NPU prefill ABI is head-major and is
transposed exactly once by the runtime loader.
"""

from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


THIS_DIR = Path(__file__).resolve().parent
PREFILL_DIR = THIS_DIR.parent / "Qwen-prefill"
if str(PREFILL_DIR) not in sys.path:
    sys.path.insert(0, str(PREFILL_DIR))
import qwen_prefill as prefill  # noqa: E402


DECODER_FORMAT = "mage-qwen3-ort-decoder-v1"
PREFILL_FORMAT = "mage-qwen-prefill-kv-v1"
HIDDEN_INPUT = "hidden"
SIN_INPUT = "position_sin"
COS_INPUT = "position_cos"
LOGITS_OUTPUT = "logits"


def past_key_name(layer: int) -> str:
    return f"past_key_values.{layer}.key"


def past_value_name(layer: int) -> str:
    return f"past_key_values.{layer}.value"


def present_key_name(layer: int) -> str:
    return f"present.{layer}.key"


def present_value_name(layer: int) -> str:
    return f"present.{layer}.value"


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))


class Qwen3DecodeLayer(nn.Module):
    """One exact Qwen3 decoder layer with explicit external past K/V.

    Projection/MLP weights are FP32 in the temporary export.  ORT dynamic
    quantization changes constant-weight MatMuls to U8-activation/S8-weight
    W8A8 operators.  Attention MatMuls remain floating point.
    """

    def __init__(
        self,
        config: prefill.QwenPrefillConfig,
        layer_id: int,
        weights: dict[str, np.ndarray],
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = int(layer_id)
        prefix = f"layers.{layer_id}."
        for suffix in prefill.LAYER_WEIGHT_SUFFIXES:
            name = suffix.replace(".", "__")
            self.register_buffer(name, _tensor(weights[prefix + suffix]))

    def _weight(self, suffix: str) -> torch.Tensor:
        return getattr(self, suffix.replace(".", "__"))

    def _rms_norm(self, value: torch.Tensor, suffix: str) -> torch.Tensor:
        variance = value.square().mean(dim=-1, keepdim=True)
        return (
            value
            * torch.rsqrt(variance + self.config.rms_norm_eps)
            * self._weight(suffix)
        )

    @staticmethod
    def _rope(
        value: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
    ) -> torch.Tensor:
        half = value.shape[-1] // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        return value * cos + rotated * sin

    def forward(
        self,
        hidden: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        position_sin: torch.Tensor,
        position_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        residual = hidden
        normalized = self._rms_norm(hidden, "input_layernorm.weight")

        query = functional.linear(normalized, self._weight("self_attn.q_proj.weight"))
        key = functional.linear(normalized, self._weight("self_attn.k_proj.weight"))
        value = functional.linear(normalized, self._weight("self_attn.v_proj.weight"))
        batch = hidden.shape[0]
        query = query.reshape(batch, 1, config.num_attention_heads, config.head_dim)
        key = key.reshape(batch, 1, config.num_key_value_heads, config.head_dim)
        value = value.reshape(batch, 1, config.num_key_value_heads, config.head_dim)
        query = self._rms_norm(query, "self_attn.q_norm.weight").transpose(1, 2)
        key = self._rms_norm(key, "self_attn.k_norm.weight").transpose(1, 2)
        query = self._rope(query, position_sin, position_cos)
        key = self._rope(key, position_sin, position_cos)

        # The public ABI is token-major FP16.  Attention is intentionally FP32;
        # the projection/MLP MatMuls are the W8A8 portion of this first engine.
        current_key = key.transpose(1, 2).to(torch.float16)
        current_value = value.to(torch.float16)
        all_key = torch.cat((past_key.float().transpose(1, 2), key), dim=2)
        all_value = torch.cat(
            (past_value.float().transpose(1, 2), value.transpose(1, 2)),
            dim=2,
        )
        # Flatten batch*KV_H and use query repeats as the MatMul M dimension.
        # This preserves efficient 3-D batched MatMul without physically
        # expanding the full cache from 8 KV heads to 32 query heads.
        past_sequence = all_key.shape[2]
        grouped_query = query.reshape(
            batch * config.num_key_value_heads,
            config.kv_repeats,
            config.head_dim,
        )
        grouped_key = all_key.reshape(
            batch * config.num_key_value_heads,
            past_sequence,
            config.head_dim,
        )
        grouped_value = all_value.reshape(
            batch * config.num_key_value_heads,
            past_sequence,
            config.head_dim,
        )
        scores = torch.matmul(grouped_query, grouped_key.transpose(-1, -2))
        scores = scores * (config.head_dim**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        attention = torch.matmul(probabilities, grouped_value)
        attention = attention.reshape(
            batch, config.num_attention_heads, config.head_dim
        ).reshape(batch, 1, config.num_attention_heads * config.head_dim)
        projected = functional.linear(
            attention, self._weight("self_attn.o_proj.weight")
        )
        hidden = residual + projected

        residual = hidden
        normalized = self._rms_norm(hidden, "post_attention_layernorm.weight")
        gate = functional.silu(
            functional.linear(normalized, self._weight("mlp.gate_proj.weight"))
        )
        up = functional.linear(normalized, self._weight("mlp.up_proj.weight"))
        down = functional.linear(gate * up, self._weight("mlp.down_proj.weight"))
        return hidden + down, current_key, current_value


class Qwen3FinalHead(nn.Module):
    """Final RMSNorm and untied LM head for one decode token."""

    def __init__(
        self,
        config: prefill.QwenPrefillConfig,
        norm_weight: np.ndarray,
        lm_head_weight: np.ndarray,
    ) -> None:
        super().__init__()
        self.eps = float(config.rms_norm_eps)
        self.register_buffer("norm_weight", _tensor(norm_weight).reshape(-1))
        self.register_buffer("lm_head_weight", _tensor(lm_head_weight))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.square().mean(dim=-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps) * self.norm_weight
        return functional.linear(hidden[:, -1], self.lm_head_weight)


def decoder_manifest(
    model_path: str | Path,
    config: prefill.QwenPrefillConfig,
    *,
    model_file: str,
    quantization: dict[str, object],
) -> dict[str, object]:
    root = Path(model_path)
    config_path = root / "config.json"
    raw = json.loads(config_path.read_text())
    text = raw.get("text_config", raw)
    index_path = root / "model.safetensors.index.json"
    fingerprint = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    if index_path.exists():
        fingerprint["weight_index_sha256"] = hashlib.sha256(
            index_path.read_bytes()
        ).hexdigest()
    return {
        "format": DECODER_FORMAT,
        "model_path": str(Path(model_path)),
        "model_file": model_file,
        "checkpoint_fingerprint": fingerprint,
        "config": asdict(config),
        "bos_token_id": int(text.get("bos_token_id", raw.get("bos_token_id", -1))),
        "eos_token_id": int(raw.get("eos_token_id", text["eos_token_id"])),
        "cache_dtype": "float16",
        "cache_layout": "BSHD",
        "cache_shape": [
            1,
            "past_sequence",
            config.num_key_value_heads,
            config.head_dim,
        ],
        "decode_sequence": 1,
        "inputs": {
            "hidden": HIDDEN_INPUT,
            "position_sin": SIN_INPUT,
            "position_cos": COS_INPUT,
            "past_key_pattern": "past_key_values.{layer}.key",
            "past_value_pattern": "past_key_values.{layer}.value",
        },
        "outputs": {
            "logits": LOGITS_OUTPUT,
            "present_key_pattern": "present.{layer}.key",
            "present_value_pattern": "present.{layer}.value",
        },
        "quantization": quantization,
    }


def expected_input_names(num_layers: int) -> set[str]:
    names = {HIDDEN_INPUT, SIN_INPUT, COS_INPUT}
    for layer in range(num_layers):
        names.add(past_key_name(layer))
        names.add(past_value_name(layer))
    return names


def expected_output_names(num_layers: int) -> set[str]:
    names = {LOGITS_OUTPUT}
    for layer in range(num_layers):
        names.add(present_key_name(layer))
        names.add(present_value_name(layer))
    return names


def validate_layers(layers: Iterable[int], count: int) -> list[int]:
    result = [int(layer) for layer in layers]
    if len(set(result)) != len(result):
        raise ValueError("layer list contains duplicates")
    if any(layer < 0 or layer >= count for layer in result):
        raise ValueError(f"layer must be in [0,{count})")
    return result
