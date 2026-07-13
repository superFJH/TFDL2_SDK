"""
Generic ViT / DINOv2 / DINOv3 TFDL2 direct builder.

This file implements the dual-graph conversion pattern described in
Doc/ViT_TFDL_Dual_Graph_Quantization.md:

  * load HuggingFace safetensors weights
  * map source weights into a small canonical ViT namespace
  * convert every parameterized Transformer Linear weight to 1x1 Conv
  * build the TFDL graph with Op.* and Convolution2 pointwise projections
  * build a PyTorch graph with the same topology on GPU to collect min/max JSON
  * map the JSON ranges back onto the TFDL symbols before SDK Quantize

The implementation intentionally keeps attention QK/AV MatMul as MatMul because
those are activation x activation products. Only parameterized Linear/MatMul
projections are rewritten to 1x1 Conv.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def _pair(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        pair = tuple(int(v) for v in value)
        if len(pair) != 2:
            raise ValueError(f"expected 2 values, got {pair}")
        return pair[0], pair[1]
    value = int(value)
    return value, value


def _factor_token_grid(seq_len: int, patch_height: int, patch_width: int) -> tuple[int, int]:
    target_ratio = float(patch_height) / float(max(patch_width, 1))
    best = (1, seq_len)
    best_score = float("inf")
    for h in range(1, int(math.sqrt(seq_len)) + 1):
        if seq_len % h != 0:
            continue
        for cand_h, cand_w in ((h, seq_len // h), (seq_len // h, h)):
            ratio = float(cand_h) / float(cand_w)
            aspect_score = abs(math.log(ratio / target_ratio))
            skinny_penalty = float(max(cand_h, cand_w)) / float(min(cand_h, cand_w))
            patch_score = (
                abs(cand_h - patch_height) / float(max(patch_height, 1))
                + abs(cand_w - patch_width) / float(max(patch_width, 1))
            )
            score = aspect_score * 4.0 + skinny_penalty * 0.1 + patch_score
            if score < best_score:
                best = (cand_h, cand_w)
                best_score = score
    return best


@dataclass
class ViTOpConfig:
    arch: str
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    num_channels: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    has_cls_token: bool = True
    num_register_tokens: int = 0
    use_position_embeddings: bool = True
    use_rope: bool = False
    rope_theta: float = 100.0
    use_gated_mlp: bool = False
    use_layer_scale: bool = True
    image_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    image_rescale_factor: float = 1.0 / 255.0

    @property
    def patch_height(self) -> int:
        return self.image_size[0] // self.patch_size[0]

    @property
    def patch_width(self) -> int:
        return self.image_size[1] // self.patch_size[1]

    @property
    def num_patches(self) -> int:
        return self.patch_height * self.patch_width

    @property
    def prefix_len(self) -> int:
        return (1 if self.has_cls_token else 0) + self.num_register_tokens

    @property
    def seq_len(self) -> int:
        return self.prefix_len + self.num_patches

    @property
    def seq_map_hw(self) -> tuple[int, int]:
        return _factor_token_grid(self.seq_len, self.patch_height, self.patch_width)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        arch: str,
        image_size: int | tuple[int, int] | None = None,
    ) -> "ViTOpConfig":
        model_path = Path(model_path)
        raw = json.loads((model_path / "config.json").read_text())
        image_mean = (0.485, 0.456, 0.406)
        image_std = (0.229, 0.224, 0.225)
        image_rescale_factor = 1.0 / 255.0
        preprocessor_path = model_path / "preprocessor_config.json"
        if preprocessor_path.exists():
            preprocessor = json.loads(preprocessor_path.read_text())
            if not preprocessor.get("do_rescale", True):
                image_rescale_factor = 1.0
            else:
                image_rescale_factor = float(preprocessor.get("rescale_factor", image_rescale_factor))
            if preprocessor.get("do_normalize", True):
                image_mean = tuple(float(v) for v in preprocessor.get("image_mean", image_mean))
                image_std = tuple(float(v) for v in preprocessor.get("image_std", image_std))
            else:
                image_mean = (0.0, 0.0, 0.0)
                image_std = (1.0, 1.0, 1.0)
        arch = arch.lower()
        resolved_image_size = _pair(image_size if image_size is not None else raw.get("image_size", 224))
        patch_size = _pair(raw.get("patch_size", 16))
        hidden_size = int(raw.get("hidden_size", raw.get("embed_dim", 768)))
        layers = int(raw.get("num_hidden_layers", raw.get("depth", 12)))
        heads = int(raw.get("num_attention_heads", raw.get("num_heads", 12)))
        mlp_ratio = float(raw.get("mlp_ratio", 4.0))
        intermediate_size = int(raw.get("intermediate_size", hidden_size * mlp_ratio))
        use_gated_mlp = bool(raw.get("use_swiglu_ffn", raw.get("use_gated_mlp", False)))
        if use_gated_mlp and "intermediate_size" not in raw:
            intermediate_size = (int(hidden_size * mlp_ratio * 2 / 3) + 7) // 8 * 8
        has_cls_token = bool(raw.get("use_cls_token", True))
        cfg = cls(
            arch=arch,
            image_size=resolved_image_size,
            patch_size=patch_size,
            num_channels=int(raw.get("num_channels", raw.get("in_chans", 3))),
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=intermediate_size,
            layer_norm_eps=float(raw.get("layer_norm_eps", raw.get("norm_eps", 1e-6))),
            qkv_bias=bool(raw.get("qkv_bias", raw.get("query_bias", True))),
            proj_bias=bool(raw.get("proj_bias", True)),
            mlp_bias=bool(raw.get("mlp_bias", raw.get("ffn_bias", True))),
            has_cls_token=has_cls_token,
            num_register_tokens=int(raw.get("num_register_tokens", raw.get("num_registers", 0))),
            use_position_embeddings=arch != "dinov3",
            use_rope=arch == "dinov3" or bool(raw.get("use_rope", False)),
            rope_theta=float(raw.get("rope_theta", raw.get("rope_freq", 100.0))),
            use_gated_mlp=use_gated_mlp,
            use_layer_scale=raw.get("layerscale_value", raw.get("layer_scale_init_value", 1.0)) is not None,
            image_mean=image_mean,
            image_std=image_std,
            image_rescale_factor=image_rescale_factor,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.image_size[0] % self.patch_size[0] or self.image_size[1] % self.patch_size[1]:
            raise ValueError("image_size must be divisible by patch_size")
        if self.use_rope and self.head_dim % 4 != 0:
            raise ValueError("DINOv3-style 2D RoPE requires head_dim divisible by 4")


def load_safetensors(model_path: str | Path) -> dict[str, np.ndarray]:
    from safetensors.torch import load_file

    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"
    single_file = model_path / "model.safetensors"
    state: dict[str, Any] = {}
    if index_file.exists():
        index = json.loads(index_file.read_text())
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(model_path / shard)))
    elif single_file.exists():
        state = load_file(str(single_file))
    else:
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors found under {model_path}")
        for shard in shards:
            state.update(load_file(str(shard)))
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}


def _as_float32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.astype(np.float32))


def _linear_to_conv1x1(weight: np.ndarray) -> np.ndarray:
    if weight.ndim == 4:
        return _as_float32(weight)
    if weight.ndim != 2:
        raise ValueError(f"expected Linear weight [out,in], got {weight.shape}")
    return _as_float32(weight[:, :, None, None])


def _fold_conv1x1_output_scale(
    weight: np.ndarray,
    bias: np.ndarray,
    scale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    weight = _linear_to_conv1x1(weight)
    bias = _as_float32(bias)
    if scale is None:
        return weight, bias
    scale = _as_float32(scale).reshape(-1)
    return _as_float32(weight * scale[:, None, None, None]), _as_float32(bias * scale)


def _first(weights: dict[str, np.ndarray], *names: str, required: bool = True) -> np.ndarray | None:
    for name in names:
        if name in weights:
            return weights[name]
    if required:
        raise KeyError("missing any of: " + ", ".join(names))
    return None


def _or_zeros(value: np.ndarray | None, shape: int | tuple[int, ...]) -> np.ndarray:
    if value is not None:
        return value
    if isinstance(shape, int):
        shape = (shape,)
    return np.zeros(shape, dtype=np.float32)


def _source_prefixes(arch: str) -> tuple[str, ...]:
    if arch == "dinov2":
        return ("", "backbone.")
    return ("", "vit.", "model.", "backbone.", "backbone.pretrained.")


def _prefix_names(prefixes: tuple[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(prefix + suffix for prefix in prefixes)


def _interpolate_position_embeddings(
    pos_embed: np.ndarray,
    config: ViTOpConfig,
) -> np.ndarray:
    if pos_embed.shape[1] == config.seq_len:
        return _as_float32(pos_embed)
    prefix = pos_embed[:, : config.prefix_len, :]
    patch = pos_embed[:, config.prefix_len :, :]
    dim = patch.shape[-1]
    src_n = patch.shape[1]
    src_h = int(round(math.sqrt(src_n)))
    src_w = src_n // src_h
    if src_h * src_w != src_n:
        raise ValueError(f"cannot infer source position grid from {src_n} patches")
    patch_t = torch.from_numpy(patch).reshape(1, src_h, src_w, dim).permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        patch_t,
        size=(config.patch_height, config.patch_width),
        mode="bicubic",
        align_corners=False,
    )
    resized = resized.permute(0, 2, 3, 1).reshape(1, config.num_patches, dim)
    return _as_float32(torch.cat([torch.from_numpy(prefix.astype(np.float32)), resized], dim=1).numpy())


def compute_2d_rope(config: ViTOpConfig) -> tuple[np.ndarray, np.ndarray]:
    periods = config.rope_theta ** (
        4.0 * np.arange(config.head_dim // 4, dtype=np.float32) / config.head_dim
    )
    rows = (np.arange(0.5, config.patch_height, dtype=np.float32) / config.patch_height) * 2.0 - 1.0
    cols = (np.arange(0.5, config.patch_width, dtype=np.float32) / config.patch_width) * 2.0 - 1.0
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    coords = np.stack([grid_r, grid_c], axis=-1).reshape(config.num_patches, 2)
    angles = 2.0 * np.pi * coords[:, :, None] / periods[None, None, :]
    angles = np.tile(angles.reshape(config.num_patches, -1), 2)
    return (
        np.sin(angles).reshape(1, 1, config.num_patches, config.head_dim).astype(np.float32),
        np.cos(angles).reshape(1, 1, config.num_patches, config.head_dim).astype(np.float32),
    )


def canonicalize_weights(
    raw: dict[str, np.ndarray],
    config: ViTOpConfig,
) -> dict[str, np.ndarray]:
    prefixes = _source_prefixes(config.arch)
    out: dict[str, np.ndarray] = {
        "patch.weight": _as_float32(_first(
            raw,
            *_prefix_names(prefixes, "embeddings.patch_embeddings.projection.weight"),
            *_prefix_names(prefixes, "embeddings.patch_embeddings.weight"),
            *_prefix_names(prefixes, "patch_embed.proj.weight"),
        )),
        "patch.bias": _as_float32(_or_zeros(_first(
            raw,
            *_prefix_names(prefixes, "embeddings.patch_embeddings.projection.bias"),
            *_prefix_names(prefixes, "embeddings.patch_embeddings.bias"),
            *_prefix_names(prefixes, "patch_embed.proj.bias"),
            required=False,
        ), config.hidden_size)),
    }

    prefix_tokens = []
    if config.has_cls_token:
        prefix_tokens.append(_as_float32(_first(raw, *_prefix_names(prefixes, "embeddings.cls_token"), *_prefix_names(prefixes, "cls_token"))))
    if config.num_register_tokens:
        reg = _first(raw, *_prefix_names(prefixes, "embeddings.register_tokens"), *_prefix_names(prefixes, "register_tokens"), required=False)
        if reg is None:
            reg = np.zeros((1, config.num_register_tokens, config.hidden_size), dtype=np.float32)
        prefix_tokens.append(_as_float32(reg))
    if prefix_tokens:
        out["prefix_tokens"] = _as_float32(np.concatenate(prefix_tokens, axis=1))
    if config.use_position_embeddings:
        pos = _first(raw, *_prefix_names(prefixes, "embeddings.position_embeddings"), *_prefix_names(prefixes, "pos_embed"))
        out["pos_embed"] = _interpolate_position_embeddings(pos, config)

    out["norm.weight"] = _as_float32(_first(raw, *_prefix_names(prefixes, "layernorm.weight"), *_prefix_names(prefixes, "norm.weight")))
    out["norm.bias"] = _as_float32(_or_zeros(_first(raw, *_prefix_names(prefixes, "layernorm.bias"), *_prefix_names(prefixes, "norm.bias"), required=False), config.hidden_size))

    for layer_id in range(config.num_hidden_layers):
        c = f"layers.{layer_id}"
        srcs = []
        for p in prefixes:
            srcs.extend(
                [
                    f"{p}encoder.layer.{layer_id}",
                    f"{p}layer.{layer_id}",
                    f"{p}blocks.{layer_id}",
                ]
            )

        def names(suffixes: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(f"{src}.{suffix}" for src in srcs for suffix in suffixes)

        out[f"{c}.norm1.weight"] = _as_float32(_first(raw, *names(("norm1.weight", "layernorm_before.weight"))))
        out[f"{c}.norm1.bias"] = _as_float32(_or_zeros(_first(raw, *names(("norm1.bias", "layernorm_before.bias")), required=False), config.hidden_size))
        out[f"{c}.norm2.weight"] = _as_float32(_first(raw, *names(("norm2.weight", "layernorm_after.weight"))))
        out[f"{c}.norm2.bias"] = _as_float32(_or_zeros(_first(raw, *names(("norm2.bias", "layernorm_after.bias")), required=False), config.hidden_size))

        qkv = _first(raw, *names(("attn.qkv.weight", "attention.qkv.weight")), required=False)
        qkv_b = _first(raw, *names(("attn.qkv.bias", "attention.qkv.bias")), required=False)
        if qkv is not None:
            q_w, k_w, v_w = np.split(qkv, 3, axis=0)
            if qkv_b is not None:
                q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
            else:
                q_b = k_b = v_b = np.zeros(config.hidden_size, dtype=np.float32)
        else:
            q_w = _first(raw, *names(("attention.attention.query.weight", "attention.query.weight", "attention.q_proj.weight")))
            k_w = _first(raw, *names(("attention.attention.key.weight", "attention.key.weight", "attention.k_proj.weight")))
            v_w = _first(raw, *names(("attention.attention.value.weight", "attention.value.weight", "attention.v_proj.weight")))
            q_b = _or_zeros(_first(raw, *names(("attention.attention.query.bias", "attention.query.bias", "attention.q_proj.bias")), required=False), config.hidden_size)
            k_b = _or_zeros(_first(raw, *names(("attention.attention.key.bias", "attention.key.bias", "attention.k_proj.bias")), required=False), config.hidden_size)
            v_b = _or_zeros(_first(raw, *names(("attention.attention.value.bias", "attention.value.bias", "attention.v_proj.bias")), required=False), config.hidden_size)

        out[f"{c}.q.weight"] = _linear_to_conv1x1(q_w)
        out[f"{c}.k.weight"] = _linear_to_conv1x1(k_w)
        out[f"{c}.v.weight"] = _linear_to_conv1x1(v_w)
        out[f"{c}.q.bias"] = _as_float32(q_b)
        out[f"{c}.k.bias"] = _as_float32(k_b)
        out[f"{c}.v.bias"] = _as_float32(v_b)

        ls1 = _first(raw, *names(("layer_scale1.lambda1", "ls1.gamma")), required=False)
        ls2 = _first(raw, *names(("layer_scale2.lambda1", "ls2.gamma")), required=False)

        proj_w = _first(raw, *names(("attention.output.dense.weight", "attention.output.weight", "attention.o_proj.weight", "attn.proj.weight")))
        proj_b = _or_zeros(_first(raw, *names(("attention.output.dense.bias", "attention.output.bias", "attention.o_proj.bias", "attn.proj.bias")), required=False), config.hidden_size)
        out[f"{c}.proj.weight"], out[f"{c}.proj.bias"] = _fold_conv1x1_output_scale(proj_w, proj_b, ls1)

        if config.use_gated_mlp:
            gate_w = _first(raw, *names(("mlp.gate_proj.weight",)), required=False)
            up_w = _first(raw, *names(("mlp.up_proj.weight", "mlp.weights_in.weight")))
            if gate_w is None and up_w.shape[0] == config.intermediate_size * 2:
                gate_w, up_w = np.split(up_w, 2, axis=0)
            gate_b = _first(raw, *names(("mlp.gate_proj.bias",)), required=False)
            up_b = _first(raw, *names(("mlp.up_proj.bias", "mlp.weights_in.bias")), required=False)
            if gate_b is None and up_b is not None and up_b.shape[0] == config.intermediate_size * 2:
                gate_b, up_b = np.split(up_b, 2, axis=0)
            out[f"{c}.gate.weight"] = _linear_to_conv1x1(gate_w)
            out[f"{c}.gate.bias"] = _as_float32(gate_b if gate_b is not None else np.zeros(config.intermediate_size, dtype=np.float32))
            out[f"{c}.up.weight"] = _linear_to_conv1x1(up_w)
            out[f"{c}.up.bias"] = _as_float32(up_b if up_b is not None else np.zeros(config.intermediate_size, dtype=np.float32))
            down_w = _first(raw, *names(("mlp.down_proj.weight", "mlp.weights_out.weight")))
            down_b = _or_zeros(_first(raw, *names(("mlp.down_proj.bias", "mlp.weights_out.bias")), required=False), config.hidden_size)
        else:
            fc1_w = _first(raw, *names(("mlp.fc1.weight", "intermediate.dense.weight", "mlp.up_proj.weight")))
            fc1_b = _or_zeros(_first(raw, *names(("mlp.fc1.bias", "intermediate.dense.bias", "mlp.up_proj.bias")), required=False), config.intermediate_size)
            out[f"{c}.fc1.weight"] = _linear_to_conv1x1(fc1_w)
            out[f"{c}.fc1.bias"] = _as_float32(fc1_b)
            down_w = _first(raw, *names(("mlp.fc2.weight", "output.dense.weight", "mlp.down_proj.weight")))
            down_b = _or_zeros(_first(raw, *names(("mlp.fc2.bias", "output.dense.bias", "mlp.down_proj.bias")), required=False), config.hidden_size)
        out[f"{c}.fc2.weight"], out[f"{c}.fc2.bias"] = _fold_conv1x1_output_scale(down_w, down_b, ls2)

    if config.use_rope:
        out["rope_sin"], out["rope_cos"] = compute_2d_rope(config)
    return out


class RangeCollector:
    def __init__(self) -> None:
        self.ranges: dict[str, tuple[float, float]] = {}

    def record(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach()
        qmin = float(detached.min().cpu())
        qmax = float(detached.max().cpu())
        if name in self.ranges:
            old_min, old_max = self.ranges[name]
            qmin = min(old_min, qmin)
            qmax = max(old_max, qmax)
        self.ranges[name] = (qmin, qmax)
        return tensor

    def to_json(self) -> dict[str, dict[str, float]]:
        return {name: {"min": qmin, "max": qmax} for name, (qmin, qmax) in sorted(self.ranges.items())}


class TorchViTOpGraph(torch.nn.Module):
    def __init__(self, config: ViTOpConfig, weights: dict[str, np.ndarray], device: torch.device):
        super().__init__()
        self.config = config
        self.weights = {k: torch.from_numpy(v).to(device) for k, v in weights.items()}
        self.collector = RangeCollector()

    def _w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def _record(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return self.collector.record(name, tensor)

    def _pointwise(self, x: torch.Tensor, prefix: str, out_channels: int, tag: str) -> torch.Tensor:
        seq_h, seq_w = self.config.seq_map_hw
        x4 = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], seq_h, seq_w)
        y4 = F.conv2d(x4, self._w(f"{prefix}.weight"), self._w(f"{prefix}.bias"))
        y = y4.reshape(x.shape[0], out_channels, self.config.seq_len).transpose(1, 2)
        return self._record(tag, y)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        sin = self._w("rope_sin")
        cos = self._w("rope_cos")

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = torch.chunk(x, 2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_prefix, q_patch = q[:, :, : cfg.prefix_len], q[:, :, cfg.prefix_len :]
        k_prefix, k_patch = k[:, :, : cfg.prefix_len], k[:, :, cfg.prefix_len :]
        q_patch = q_patch * cos + rotate_half(q_patch) * sin
        k_patch = k_patch * cos + rotate_half(k_patch) * sin
        return torch.cat([q_prefix, q_patch], dim=2), torch.cat([k_prefix, k_patch], dim=2)

    def _block(self, hidden: torch.Tensor, layer_id: int) -> torch.Tensor:
        cfg = self.config
        c = f"layers.{layer_id}"
        normed = F.layer_norm(hidden, (cfg.hidden_size,), self._w(f"{c}.norm1.weight"), self._w(f"{c}.norm1.bias"), cfg.layer_norm_eps)
        normed = self._record(f"{c}.norm1", normed)
        q = self._pointwise(normed, f"{c}.q", cfg.hidden_size, f"{c}.q")
        k = self._pointwise(normed, f"{c}.k", cfg.hidden_size, f"{c}.k")
        v = self._pointwise(normed, f"{c}.v", cfg.hidden_size, f"{c}.v")
        bsz, seq_len, _ = q.shape
        q = q.reshape(bsz, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        if cfg.use_rope:
            q, k = self._apply_rope(q, k)
            q = self._record(f"{c}.q_rope", q)
            k = self._record(f"{c}.k_rope", k)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(cfg.head_dim))
        probs = torch.softmax(self._record(f"{c}.attn_scores", scores), dim=-1)
        probs = self._record(f"{c}.attn_probs", probs)
        attn = torch.matmul(probs, v).transpose(1, 2).reshape(bsz, seq_len, cfg.hidden_size)
        attn = self._record(f"{c}.attn", attn)
        proj = self._pointwise(attn, f"{c}.proj", cfg.hidden_size, f"{c}.proj")
        hidden = self._record(f"{c}.resid1", hidden + proj)
        normed2 = F.layer_norm(hidden, (cfg.hidden_size,), self._w(f"{c}.norm2.weight"), self._w(f"{c}.norm2.bias"), cfg.layer_norm_eps)
        normed2 = self._record(f"{c}.norm2", normed2)
        if cfg.use_gated_mlp:
            gate = self._pointwise(normed2, f"{c}.gate", cfg.intermediate_size, f"{c}.gate")
            gate = self._record(f"{c}.gate_act", F.silu(gate))
            up = self._pointwise(normed2, f"{c}.up", cfg.intermediate_size, f"{c}.up")
            mlp = self._record(f"{c}.mlp_mid", gate * up)
        else:
            mlp = F.gelu(self._pointwise(normed2, f"{c}.fc1", cfg.intermediate_size, f"{c}.fc1"))
            mlp = self._record(f"{c}.mlp_mid", mlp)
        mlp = self._pointwise(mlp, f"{c}.fc2", cfg.hidden_size, f"{c}.fc2")
        return self._record(f"{c}.resid2", hidden + mlp)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        hidden = F.conv2d(pixel_values, self._w("patch.weight"), self._w("patch.bias"), stride=cfg.patch_size)
        hidden = self._record("patch.conv", hidden)
        hidden = hidden.reshape(pixel_values.shape[0], cfg.hidden_size, cfg.num_patches).transpose(1, 2)
        if cfg.prefix_len:
            prefix = self._w("prefix_tokens").expand(pixel_values.shape[0], -1, -1)
            hidden = torch.cat((prefix, hidden), dim=1)
        if cfg.use_position_embeddings:
            hidden = hidden + self._w("pos_embed")
        hidden = self._record("tokens", hidden)
        for layer_id in range(cfg.num_hidden_layers):
            hidden = self._block(hidden, layer_id)
        hidden = F.layer_norm(hidden, (cfg.hidden_size,), self._w("norm.weight"), self._w("norm.bias"), cfg.layer_norm_eps)
        hidden = self._record("final_norm", hidden)
        cls = hidden[:, 0]
        return self._record("output.cls", cls), hidden


def _load_calibration_samples(
    config: ViTOpConfig,
    calib_dir: str | Path | None,
    count: int,
    device: torch.device,
) -> list[torch.Tensor]:
    samples: list[torch.Tensor] = []
    if calib_dir:
        for path in sorted(Path(calib_dir).glob("*")):
            if len(samples) >= count:
                break
            if path.suffix == ".npy":
                arr = np.load(path).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[None, ...]
                samples.append(torch.from_numpy(arr).to(device))
            elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise ImportError("PIL is required to use image calibration files") from exc
                img = Image.open(path).convert("RGB").resize((config.image_size[1], config.image_size[0]))
                arr = np.asarray(img).astype(np.float32) * float(config.image_rescale_factor)
                mean = np.array(config.image_mean, dtype=np.float32)
                std = np.array(config.image_std, dtype=np.float32)
                arr = ((arr - mean) / std).transpose(2, 0, 1)[None, ...]
                samples.append(torch.from_numpy(arr).to(device))
    if not samples:
        for _ in range(count):
            samples.append(
                torch.randn(
                    1,
                    config.num_channels,
                    config.image_size[0],
                    config.image_size[1],
                    device=device,
                )
                * 0.02
            )
    return samples


def _preprocess_raw_nchw(config: ViTOpConfig, raw_nchw: np.ndarray) -> np.ndarray:
    mean = np.array(config.image_mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.array(config.image_std, dtype=np.float32).reshape(1, -1, 1, 1)
    return _as_float32((raw_nchw.astype(np.float32) * float(config.image_rescale_factor) - mean) / std)


def _depreprocess_to_raw_nchw(config: ViTOpConfig, normalized_nchw: np.ndarray) -> np.ndarray:
    mean = np.array(config.image_mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.array(config.image_std, dtype=np.float32).reshape(1, -1, 1, 1)
    return _as_float32((normalized_nchw.astype(np.float32) * std + mean) / float(config.image_rescale_factor))


def _load_compare_samples(
    config: ViTOpConfig,
    calib_dir: str | Path | None,
    count: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, np.ndarray]]:
    samples: list[tuple[torch.Tensor, np.ndarray]] = []
    if calib_dir:
        for path in sorted(Path(calib_dir).glob("*")):
            if len(samples) >= count:
                break
            if path.suffix == ".npy":
                arr = np.load(path).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[None, ...]
                if arr.max() > 2.0 and arr.min() >= 0.0:
                    raw = _as_float32(arr)
                    normalized = _preprocess_raw_nchw(config, raw)
                else:
                    normalized = _as_float32(arr)
                    raw = _depreprocess_to_raw_nchw(config, normalized)
                samples.append((torch.from_numpy(normalized).to(device), raw))
            elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise ImportError("PIL is required to use image calibration files") from exc
                img = Image.open(path).convert("RGB").resize((config.image_size[1], config.image_size[0]))
                raw = np.asarray(img).astype(np.float32).transpose(2, 0, 1)[None, ...]
                normalized = _preprocess_raw_nchw(config, raw)
                samples.append((torch.from_numpy(normalized).to(device), _as_float32(raw)))
    if not samples:
        normalized_samples = _load_calibration_samples(config, None, count, device)
        for sample in normalized_samples:
            normalized = sample.detach().cpu().numpy().astype(np.float32)
            samples.append((sample, _depreprocess_to_raw_nchw(config, normalized)))
    return samples


def collect_minmax_json(
    config: ViTOpConfig,
    weights: dict[str, np.ndarray],
    output_json: str | Path,
    calib_dir: str | Path | None = None,
    num_samples: int = 8,
    device_name: str | None = None,
) -> dict[str, dict[str, float]]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TorchViTOpGraph(config, weights, device).eval()
    samples = _load_calibration_samples(config, calib_dir, num_samples, device)
    with torch.no_grad():
        for sample in samples:
            model(sample.float())
    ranges = model.collector.to_json()
    Path(output_json).write_text(json.dumps(ranges, indent=2, sort_keys=True))
    return ranges


def _extract_last_hidden_state(output: Any) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"cannot find last_hidden_state in reference output: {type(output)!r}")


def _run_reference_model(model: torch.nn.Module, sample: torch.Tensor) -> torch.Tensor:
    try:
        output = model(pixel_values=sample, interpolate_pos_encoding=True)
    except TypeError:
        try:
            output = model(pixel_values=sample)
        except TypeError:
            output = model(sample)
    return _extract_last_hidden_state(output).float()


def _load_reference_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("--compare-reference/--compare-tfdl-fp requires transformers") from exc

    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    return model.eval().to(device=device, dtype=torch.float32)


def _cosine_np(ref: np.ndarray, got: np.ndarray) -> float:
    ref_flat = np.asarray(ref, dtype=np.float32).reshape(-1)
    got_flat = np.asarray(got, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(got_flat))
    if denom == 0.0:
        return 1.0 if np.array_equal(ref_flat, got_flat) else 0.0
    return float(np.dot(ref_flat, got_flat) / denom)


def _compare_np(name: str, ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    if ref.shape != got.shape:
        raise ValueError(f"{name} shape mismatch: reference={ref.shape}, converted={got.shape}")
    diff = got - ref
    return {
        "shape": list(ref.shape),
        "cos": _cosine_np(ref, got),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _print_compare_line(sample_id: int, backend: str, stats: dict[str, dict[str, Any]]) -> None:
    cls = stats["cls"]
    tokens = stats["tokens"]
    print(
        f"[COMPARE] sample={sample_id} backend={backend} "
        f"cls_cos={cls['cos']:.8f} cls_max_abs={cls['max_abs']:.6g} "
        f"tokens_cos={tokens['cos']:.8f} tokens_max_abs={tokens['max_abs']:.6g}"
    )


def compare_with_reference(
    config: ViTOpConfig,
    weights: dict[str, np.ndarray],
    model_path: str | Path,
    output_json: str | Path | None = None,
    calib_dir: str | Path | None = None,
    num_samples: int = 1,
    device_name: str | None = None,
    compare_torch_op: bool = True,
    compare_tfdl_fp: bool = False,
    addon_path: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    result: dict[str, Any] = {
        "device": str(device),
        "samples": [],
    }
    try:
        reference = _load_reference_model(model_path, device)
        torch_op = TorchViTOpGraph(config, weights, device).eval() if compare_torch_op else None
        samples = _load_compare_samples(config, calib_dir, max(1, num_samples), device)

        tfdl_executor = None
        if compare_tfdl_fp:
            if addon_path is not None:
                _load_addon_if_needed(config, addon_path)
            _, tfdl_executor, _, _, _ = build_vit_tfdl_graph(config, weights, create_executor=True)

        with torch.no_grad():
            for sample_id, (sample, tfdl_input) in enumerate(samples):
                sample = sample.float()
                ref_tokens_t = _run_reference_model(reference, sample)
                ref_cls = ref_tokens_t[:, 0].detach().cpu().numpy()
                ref_tokens = ref_tokens_t.detach().cpu().numpy()
                sample_result: dict[str, Any] = {"sample": sample_id}

                if torch_op is not None:
                    got_cls_t, got_tokens_t = torch_op(sample)
                    torch_stats = {
                        "cls": _compare_np("torch_op.cls", ref_cls, got_cls_t.detach().cpu().numpy()),
                        "tokens": _compare_np("torch_op.tokens", ref_tokens, got_tokens_t.detach().cpu().numpy()),
                    }
                    sample_result["torch_op"] = torch_stats
                    _print_compare_line(sample_id, "torch_op", torch_stats)

                if tfdl_executor is not None:
                    if sample.shape[0] != 1:
                        raise ValueError("--compare-tfdl-fp currently expects batch=1 samples")
                    inputs = tfdl_executor.GetInputs()
                    if len(inputs) != 1:
                        raise RuntimeError(f"expected one TFDL input, got {len(inputs)}")
                    inputs[0].fromNumpy(np.ascontiguousarray(tfdl_input.astype(np.float32)))
                    outputs = tfdl_executor()
                    got_cls = outputs[0].toNumpy().astype(np.float32)
                    got_tokens = outputs[1].toNumpy().astype(np.float32)
                    tfdl_stats = {
                        "cls": _compare_np("tfdl_fp.cls", ref_cls, got_cls),
                        "tokens": _compare_np("tfdl_fp.tokens", ref_tokens, got_tokens),
                    }
                    sample_result["tfdl_fp"] = tfdl_stats
                    _print_compare_line(sample_id, "tfdl_fp", tfdl_stats)

                result["samples"].append(sample_result)

        if output_json:
            Path(output_json).write_text(json.dumps(result, indent=2, sort_keys=True))
        return result
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _load_range_json(path: str | Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, tuple[float, float]] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            out[name] = (float(value["min"]), float(value["max"]))
        else:
            out[name] = (float(value[0]), float(value[1]))
    return out


def annotate_minmax_json_with_symbol_map(path: str | Path, symbol_map: dict[str, str]) -> None:
    path = Path(path)
    raw = json.loads(path.read_text())
    annotated: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            item = dict(value)
        else:
            item = {"min": float(value[0]), "max": float(value[1])}
        if name in symbol_map:
            item["tfdl_name"] = symbol_map[name]
        annotated[name] = item
    path.write_text(json.dumps(annotated, indent=2, sort_keys=True))


def _tokens_to_conv1d(hidden, config: ViTOpConfig):
    from TFDL2 import Op

    seq_h, seq_w = config.seq_map_hw
    return Op.Reshape(Op.Transpose(hidden, (0, 2, 1)), (1, config.hidden_size, seq_h, seq_w))


def _conv1d_to_tokens(hidden_4d, channels: int, config: ViTOpConfig):
    from TFDL2 import Op

    return Op.Transpose(Op.Reshape(hidden_4d, (1, channels, config.seq_len)), (0, 2, 1))


def _mark(symbol_map: dict[str, str], tag: str, symbol):
    symbol_map[tag] = str(symbol)
    return symbol


def _pointwise_4d_op(ctx, hidden_4d, weight_name: str, bias_name: str, out_channels: int, symbol_map: dict[str, str], tag: str):
    from TFDL2 import Op

    out_4d = Op.Convolution2(
        hidden_4d,
        ctx.GetParamSymbol(weight_name),
        ctx.GetParamSymbol(bias_name),
        kernel=1,
        pad=0,
        stride=1,
        dilation=1,
        outChannel=out_channels,
        group=1,
    )
    return _mark(symbol_map, tag, out_4d)


def _qkv_conv_to_heads(hidden_4d, config: ViTOpConfig):
    from TFDL2 import Op

    heads = Op.Reshape(hidden_4d, (1, config.num_attention_heads, config.head_dim, -1))
    return Op.Transpose(heads, (0, 1, 3, 2))


def _attention_to_conv1d(attn_3d, config: ViTOpConfig):
    from TFDL2 import Op

    seq_h, seq_w = config.seq_map_hw
    attn_4d = Op.Reshape(attn_3d, (1, config.num_attention_heads, config.seq_len, config.head_dim))
    attn_4d = Op.Transpose(attn_4d, (0, 1, 3, 2))
    return Op.Reshape(attn_4d, (1, config.hidden_size, seq_h, seq_w))


def _apply_rope_op(q, k, rope_sin, rope_cos, layer_id: int):
    from TFDL2 import Op

    out = Op.Custom((q, k, rope_sin, rope_cos), (f"vit_q_rope_{layer_id}", f"vit_k_rope_{layer_id}"), "ApplyRope", "{}")
    return out[0], out[1]


def _build_block_op(ctx, hidden, config: ViTOpConfig, layer_id: int, symbol_map: dict[str, str], rope_sin=None, rope_cos=None):
    from TFDL2 import Op

    c = f"layers.{layer_id}"
    normed = Op.LayerNorm2(hidden, ctx.GetParamSymbol(f"{c}.norm1.weight"), ctx.GetParamSymbol(f"{c}.norm1.bias"), axis=-1)
    normed = _mark(symbol_map, f"{c}.norm1", normed)
    normed_4d = _tokens_to_conv1d(normed, config)
    q = _qkv_conv_to_heads(_pointwise_4d_op(ctx, normed_4d, f"{c}.q.weight", f"{c}.q.bias", config.hidden_size, symbol_map, f"{c}.q"), config)
    k = _qkv_conv_to_heads(_pointwise_4d_op(ctx, normed_4d, f"{c}.k.weight", f"{c}.k.bias", config.hidden_size, symbol_map, f"{c}.k"), config)
    v = _qkv_conv_to_heads(_pointwise_4d_op(ctx, normed_4d, f"{c}.v.weight", f"{c}.v.bias", config.hidden_size, symbol_map, f"{c}.v"), config)
    if config.use_rope:
        q, k = _apply_rope_op(q, k, rope_sin, rope_cos, layer_id)
        _mark(symbol_map, f"{c}.q_rope", q)
        _mark(symbol_map, f"{c}.k_rope", k)
    q3 = Op.Reshape(q, (config.num_attention_heads, config.seq_len, config.head_dim))
    k3 = Op.Transpose(Op.Reshape(k, (config.num_attention_heads, config.seq_len, config.head_dim)), (0, 2, 1))
    v3 = Op.Reshape(v, (config.num_attention_heads, config.seq_len, config.head_dim))
    scores = _mark(symbol_map, f"{c}.attn_scores", Op.Mul(Op.MatMul(q3, k3, transA=False, transB=False), 1.0 / math.sqrt(config.head_dim)))
    probs = _mark(symbol_map, f"{c}.attn_probs", Op.Softmax(scores, axis=2))
    attn = Op.MatMul(probs, v3, transA=False, transB=False)
    attn_4d = _mark(symbol_map, f"{c}.attn", _attention_to_conv1d(attn, config))
    proj = _conv1d_to_tokens(
        _pointwise_4d_op(ctx, attn_4d, f"{c}.proj.weight", f"{c}.proj.bias", config.hidden_size, symbol_map, f"{c}.proj"),
        config.hidden_size,
        config,
    )
    hidden = _mark(symbol_map, f"{c}.resid1", Op.Add(hidden, proj))
    normed2 = _mark(symbol_map, f"{c}.norm2", Op.LayerNorm2(hidden, ctx.GetParamSymbol(f"{c}.norm2.weight"), ctx.GetParamSymbol(f"{c}.norm2.bias"), axis=-1))
    normed2_4d = _tokens_to_conv1d(normed2, config)
    if config.use_gated_mlp:
        gate = _pointwise_4d_op(ctx, normed2_4d, f"{c}.gate.weight", f"{c}.gate.bias", config.intermediate_size, symbol_map, f"{c}.gate")
        gate = _mark(symbol_map, f"{c}.gate_act", Op.Swish(gate))
        up = _pointwise_4d_op(ctx, normed2_4d, f"{c}.up.weight", f"{c}.up.bias", config.intermediate_size, symbol_map, f"{c}.up")
        mlp = _mark(symbol_map, f"{c}.mlp_mid", Op.Mul(gate, up))
    else:
        mlp = _pointwise_4d_op(ctx, normed2_4d, f"{c}.fc1.weight", f"{c}.fc1.bias", config.intermediate_size, symbol_map, f"{c}.fc1")
        mlp = _mark(symbol_map, f"{c}.mlp_mid", Op.GeLU(mlp))
    mlp = _conv1d_to_tokens(
        _pointwise_4d_op(ctx, mlp, f"{c}.fc2.weight", f"{c}.fc2.bias", config.hidden_size, symbol_map, f"{c}.fc2"),
        config.hidden_size,
        config,
    )
    return _mark(symbol_map, f"{c}.resid2", Op.Add(hidden, mlp))


def build_vit_tfdl_graph(
    config: ViTOpConfig,
    weights: dict[str, np.ndarray],
    range_json: str | Path | None = None,
    create_executor: bool = False,
):
    from TFDL2 import TFContext, TFExecutor, Op
    from TFDL2.Common import TFDataType

    ctx = TFContext(f"{config.arch}_vit_op")
    ctx.RegisterParamToContext(**weights)
    symbol_map: dict[str, str] = {}
    with ctx:
        input_scale = tuple(
            float(v)
            for v in (float(config.image_rescale_factor) / np.array(config.image_std, dtype=np.float32))
        )
        input_mean = tuple(float(v) / float(config.image_rescale_factor) for v in config.image_mean)
        pixel_values = Op.Placeholder2(
            ctx,
            shape=(1, config.num_channels, config.image_size[0], config.image_size[1]),
            outDatatype=TFDataType.TFDL_FLOAT,
            scale=input_scale,
            mean=input_mean,
        )
        input_name = str(pixel_values)
        hidden = Op.Convolution2(
            pixel_values,
            ctx.GetParamSymbol("patch.weight"),
            ctx.GetParamSymbol("patch.bias"),
            kernel=config.patch_size,
            pad=0,
            stride=config.patch_size,
            dilation=1,
            outChannel=config.hidden_size,
            group=1,
        )
        _mark(symbol_map, "patch.conv", hidden)
        hidden = Op.Transpose(Op.Reshape(hidden, (1, config.hidden_size, config.num_patches)), (0, 2, 1))
        if config.prefix_len:
            hidden = Op.Concat((ctx.GetParamSymbol("prefix_tokens"), hidden), axis=1)
        if config.use_position_embeddings:
            hidden = Op.Add(hidden, ctx.GetParamSymbol("pos_embed"))
        hidden = _mark(symbol_map, "tokens", hidden)
        rope_sin = ctx.GetParamSymbol("rope_sin") if config.use_rope else None
        rope_cos = ctx.GetParamSymbol("rope_cos") if config.use_rope else None
        for layer_id in range(config.num_hidden_layers):
            hidden = _build_block_op(ctx, hidden, config, layer_id, symbol_map, rope_sin, rope_cos)
        hidden = _mark(symbol_map, "final_norm", Op.LayerNorm2(hidden, ctx.GetParamSymbol("norm.weight"), ctx.GetParamSymbol("norm.bias"), axis=-1))
        cls = Op.Gather2(hidden, 0, 1)
        cls = _mark(symbol_map, "output.cls", Op.Reshape(cls, (1, config.hidden_size)))
        output_names = [str(cls), str(hidden)]
    ctx.SetOutputs(output_names)

    if range_json:
        ranges = _load_range_json(range_json)
        for tag, actual in symbol_map.items():
            if tag in ranges:
                qmin, qmax = ranges[tag]
                if not ctx.AddInt8Config(actual, float(qmax), float(qmin)):
                    raise RuntimeError(f"failed to add int8 config for {tag} -> {actual}")

    executor = None
    if create_executor:
        executor = TFExecutor(ctx, {"UseHardware": False, "FrugalMode": True})
    return ctx, executor, [input_name], output_names, symbol_map


def dump_context(ctx: Any, output: str | Path) -> None:
    output = str(output)
    if output.endswith(".fb"):
        output = output[:-3]
    ctx.Dump(output)


def quantize_with_ranges(ctx: Any, input_names: list[str], output_names: list[str], output: str | Path) -> None:
    from TFDL2 import TFCalibration, CalibrationMode
    from TFDL2.Common import TFDataType

    calib = TFCalibration(ctx, CalibrationMode.Naive, {"UseHardware": False, "FrugalMode": True})
    calib.Quantize(
        {name: TFDataType.TFDL_UINT8 for name in input_names},
        stopquanttensors=tuple(output_names),
        MergeConcate=False,
        Perchannel=True,
    )
    ctx.SetOutputs(output_names)
    dump_context(ctx, output)


def build_arg_parser(default_arch: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ViT/DINO TFDL2 Op graph and optional GPU min/max JSON")
    parser.add_argument("--arch", default=default_arch or "vit", choices=("vit", "dinov2", "dinov3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-size", type=int, nargs='+', default=None)
    parser.add_argument("--calib-dir", default=None)
    parser.add_argument("--num-calib", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default prefers CUDA.")
    parser.add_argument("--compare-reference", action="store_true", help="Compare PyTorch Op-equivalent graph with the original transformers model")
    parser.add_argument("--compare-tfdl-fp", action="store_true", help="Compare the built TFDL float executor with the original transformers model")
    parser.add_argument("--compare-json", default=None, help="Optional path to save cosine/max error comparison stats")
    parser.add_argument("--dump-minmax-json", default=None)
    parser.add_argument("--range-json", default=None, help="Existing min/max JSON to map to TFDL AddInt8Config")
    parser.add_argument("--dump-fb", default=None)
    parser.add_argument("--dump-quant-fb", default=None)
    parser.add_argument("--dump-symbol-map", default=None)
    parser.add_argument(
        "--addon-path",
        default=str(Path(__file__).resolve().parents[3] / "AddonOps" / "build" / "libTFDLAddOn.so"),
        help="Custom op shared library path. Required for DINOv3 ApplyRope graph build/run.",
    )
    return parser


def _load_addon_if_needed(config: ViTOpConfig, addon_path: str | Path) -> None:
    if not config.use_rope:
        return
    addon_path = Path(addon_path)
    if not addon_path.exists():
        raise FileNotFoundError(f"DINOv3 RoPE graph needs custom op library: {addon_path}")
    from TFDL2.utils import LoadCustomOp

    LoadCustomOp(str(addon_path))


def main(argv: list[str] | None = None, default_arch: str | None = None) -> None:
    args = build_arg_parser(default_arch).parse_args(argv)
    config = ViTOpConfig.from_model_path(args.model_path, args.arch, image_size=args.image_size)
    raw = load_safetensors(args.model_path)
    weights = canonicalize_weights(raw, config)
    range_json = args.range_json
    if args.compare_reference or args.compare_tfdl_fp:
        compare_with_reference(
            config,
            weights,
            args.model_path,
            output_json=args.compare_json,
            calib_dir=args.calib_dir,
            num_samples=args.num_calib,
            device_name=args.device,
            compare_torch_op=args.compare_reference,
            compare_tfdl_fp=args.compare_tfdl_fp,
            addon_path=args.addon_path,
        )
    if args.dump_minmax_json:
        collect_minmax_json(
            config,
            weights,
            args.dump_minmax_json,
            calib_dir=args.calib_dir,
            num_samples=args.num_calib,
            device_name=args.device,
        )
        range_json = args.dump_minmax_json
    if args.dump_fb or args.dump_quant_fb or args.dump_symbol_map:
        _load_addon_if_needed(config, args.addon_path)
        ctx, _, input_names, output_names, symbol_map = build_vit_tfdl_graph(config, weights, range_json=range_json)
        if args.dump_minmax_json:
            annotate_minmax_json_with_symbol_map(args.dump_minmax_json, symbol_map)
        if args.dump_symbol_map:
            Path(args.dump_symbol_map).write_text(json.dumps(symbol_map, indent=2, sort_keys=True))
        if args.dump_fb:
            dump_context(ctx, args.dump_fb)
        if args.dump_quant_fb:
            if not range_json:
                raise ValueError("--dump-quant-fb requires --range-json or --dump-minmax-json")
            quantize_with_ranges(ctx, input_names, output_names, args.dump_quant_fb)
    print(
        f"[OK] {args.arch} graph flow prepared: image={config.image_size}, "
        f"seq_len={config.seq_len}, seq_map_hw={config.seq_map_hw}, layers={config.num_hidden_layers}"
    )


if __name__ == "__main__":
    main()
