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
        if len(pair) == 1:
            return pair[0], pair[0]
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
    """Collect activation ranges without relying on SDK calibration modes.

    ``naive``/``minmax`` and ``mean`` only need the extrema of every
    observation. The clipping methods additionally keep a deterministic,
    evenly spaced sample of each tensor. This bounds host memory for large ViT
    attention tensors while still sampling every calibration image and the
    complete tensor.
    """

    METHODS = (
        "naive",
        "minmax",
        "mean",
        "entropy",
        "coverage",
        "percentile",
        "mse",
    )

    def __init__(
        self,
        method: str = "minmax",
        *,
        num_observations: int = 8,
        max_samples_per_tensor: int = 65536,
        percentile: float = 99.99,
        coverage: float = 99.99,
        token_axis_size: int | None = None,
        register_start: int = 1,
        num_register_tokens: int = 0,
        register_range_policy: str = "include",
    ) -> None:
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(f"unsupported range method {method!r}; expected one of {self.METHODS}")
        self.method = method
        self.ranges: dict[str, tuple[float, float]] = {}
        self._range_sums: dict[str, tuple[float, float, int]] = {}
        self._samples: dict[str, list[np.ndarray]] = {}
        self._samples_per_observation = max(
            256, int(max_samples_per_tensor) // max(1, int(num_observations))
        )
        self._max_samples_per_tensor = max(256, int(max_samples_per_tensor))
        self.percentile = float(percentile)
        self.coverage = float(coverage)
        self.token_axis_size = int(token_axis_size) if token_axis_size is not None else None
        self.register_start = int(register_start)
        self.num_register_tokens = int(num_register_tokens)
        if register_range_policy not in {"include", "exclude", "role-aware"}:
            raise ValueError(
                "register_range_policy must be include, exclude, or role-aware"
            )
        self.register_range_policy = register_range_policy
        self.exclude_register_tokens = (
            self.register_range_policy != "include"
            and self.token_axis_size is not None
            and self.num_register_tokens > 0
        )

    def set_fixed_range(self, name: str, qmin: float, qmax: float) -> None:
        self.ranges[name] = (float(qmin), float(qmax))

    def _exclude_register_tokens(
        self, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        if (
            not self.exclude_register_tokens
            or self.num_register_tokens <= 0
            or self.token_axis_size is None
        ):
            return tensor
        if self.register_range_policy == "role-aware" and (
            name.endswith(".k")
            or name.endswith(".k_rope")
            or name.endswith(".v")
        ):
            # Register K/V are consumed by CLS and patch queries, so clipping
            # them would corrupt attention for every useful output token.
            return tensor
        register_end = self.register_start + self.num_register_tokens
        keep = torch.cat(
            (
                torch.arange(
                    0,
                    self.register_start,
                    device=tensor.device,
                ),
                torch.arange(
                    register_end,
                    self.token_axis_size,
                    device=tensor.device,
                ),
            )
        )
        filtered = tensor
        # Token activations have one S axis. Attention score/probability tensors
        # have query-S and key-S axes; role-aware calibration excludes only
        # register queries and retains register keys.
        for axis in range(1, filtered.ndim):
            if int(filtered.shape[axis]) == self.token_axis_size:
                filtered = filtered.index_select(axis, keep)
                if (
                    self.register_range_policy == "role-aware"
                    and name.endswith(
                        (".qk_matmul", ".attn_scores", ".attn_probs")
                    )
                ):
                    break
        return filtered

    @staticmethod
    def _include_zero(qmin: float, qmax: float) -> tuple[float, float]:
        qmin = min(float(qmin), 0.0)
        qmax = max(float(qmax), 0.0)
        if not np.isfinite(qmin) or not np.isfinite(qmax):
            raise ValueError(f"non-finite calibration range: min={qmin}, max={qmax}")
        if qmin == qmax:
            epsilon = max(abs(qmin), 1.0) * 1e-6
            qmin -= epsilon
            qmax += epsilon
        return qmin, qmax

    def record(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach()
        observed = self._exclude_register_tokens(name, detached)
        observed_min = float(observed.min().cpu())
        observed_max = float(observed.max().cpu())
        qmin, qmax = observed_min, observed_max
        if name in self.ranges:
            old_min, old_max = self.ranges[name]
            qmin = min(old_min, qmin)
            qmax = max(old_max, qmax)
        self.ranges[name] = (qmin, qmax)

        sum_min, sum_max, count = self._range_sums.get(name, (0.0, 0.0, 0))
        self._range_sums[name] = (
            sum_min + observed_min,
            sum_max + observed_max,
            count + 1,
        )

        if self.method in {"entropy", "coverage", "percentile", "mse"}:
            flat = observed.reshape(-1)
            take = min(int(flat.numel()), self._samples_per_observation)
            if take:
                if take == int(flat.numel()):
                    sampled = flat
                else:
                    indices = torch.linspace(
                        0, int(flat.numel()) - 1, take, device=flat.device
                    ).long()
                    sampled = flat.index_select(0, indices)
                self._samples.setdefault(name, []).append(
                    sampled.float().cpu().numpy().astype(np.float32, copy=False)
                )
        return tensor

    def _values(self, name: str) -> np.ndarray:
        chunks = self._samples.get(name, ())
        if not chunks:
            qmin, qmax = self.ranges[name]
            return np.asarray((qmin, qmax), dtype=np.float32)
        values = np.concatenate(chunks)
        if values.size > self._max_samples_per_tensor:
            indices = np.linspace(
                0, values.size - 1, self._max_samples_per_tensor, dtype=np.int64
            )
            values = values[indices]
        return values[np.isfinite(values)]

    @staticmethod
    def _entropy_abs_threshold(values: np.ndarray, bins: int = 2048, quant_bins: int = 128) -> float:
        """TensorRT-style KL/entropy threshold over absolute magnitudes."""
        absolute = np.abs(values.astype(np.float64, copy=False))
        maximum = float(absolute.max(initial=0.0))
        if maximum <= 0.0:
            return 0.0
        hist, edges = np.histogram(absolute, bins=bins, range=(0.0, maximum))
        hist = hist.astype(np.float64)
        best_kl = float("inf")
        best_bin = bins
        # Skip the degenerate threshold where one histogram bin maps exactly
        # to one quantized bin and KL is mechanically zero.
        # Searching every one of the 2048 bins is unnecessarily expensive for
        # hundreds of transformer activations.  A dense 128-candidate grid is
        # stable in practice and keeps calibration time bounded.
        candidate_bins = np.unique(
            np.linspace(quant_bins + 1, bins, 128, dtype=np.int64)
        )
        for threshold_bin_value in candidate_bins:
            threshold_bin = int(threshold_bin_value)
            reference = hist[:threshold_bin].copy()
            if threshold_bin < bins:
                reference[-1] += hist[threshold_bin:].sum()
            total = reference.sum()
            if total <= 0.0:
                continue
            reference /= total

            quantized = np.zeros(quant_bins, dtype=np.float64)
            boundaries = np.linspace(0, threshold_bin, quant_bins + 1, dtype=np.int64)
            for index in range(quant_bins):
                start, end = int(boundaries[index]), int(boundaries[index + 1])
                if end <= start:
                    end = min(start + 1, threshold_bin)
                quantized[index] = reference[start:end].sum()

            expanded = np.zeros(threshold_bin, dtype=np.float64)
            for index in range(quant_bins):
                start, end = int(boundaries[index]), int(boundaries[index + 1])
                if end <= start:
                    end = min(start + 1, threshold_bin)
                nonzero = reference[start:end] > 0
                nonzero_count = int(nonzero.sum())
                if nonzero_count:
                    expanded[start:end][nonzero] = quantized[index] / nonzero_count

            mask = reference > 0
            if np.any(expanded[mask] <= 0):
                continue
            kl = float(np.sum(reference[mask] * np.log(reference[mask] / expanded[mask])))
            if kl < best_kl:
                best_kl = kl
                best_bin = threshold_bin
        return float(edges[best_bin])

    @staticmethod
    def _shortest_coverage_interval(values: np.ndarray, coverage_percent: float) -> tuple[float, float]:
        ordered = np.sort(values.astype(np.float64, copy=False))
        if ordered.size < 2:
            value = float(ordered[0]) if ordered.size else 0.0
            return value, value
        keep = min(
            ordered.size,
            max(2, int(math.ceil(ordered.size * coverage_percent / 100.0))),
        )
        if keep == ordered.size:
            return float(ordered[0]), float(ordered[-1])
        widths = ordered[keep - 1 :] - ordered[: ordered.size - keep + 1]
        start = int(np.argmin(widths))
        return float(ordered[start]), float(ordered[start + keep - 1])

    @staticmethod
    def _mse_range(values: np.ndarray) -> tuple[float, float]:
        values64 = values.astype(np.float64, copy=False)
        best = (float(values64.min()), float(values64.max()))
        best_error = float("inf")
        for coverage in (95.0, 97.5, 99.0, 99.5, 99.9, 99.95, 99.99, 100.0):
            tail = (100.0 - coverage) * 0.5
            qmin, qmax = np.percentile(values64, (tail, 100.0 - tail))
            qmin, qmax = RangeCollector._include_zero(float(qmin), float(qmax))
            scale = (qmax - qmin) / 255.0
            quantized = np.clip(np.rint((values64 - qmin) / scale), 0.0, 255.0)
            restored = quantized * scale + qmin
            error = float(np.mean(np.square(restored - values64)))
            if error < best_error:
                best_error = error
                best = (qmin, qmax)
        return best

    def resolved_ranges(self) -> dict[str, tuple[float, float]]:
        resolved: dict[str, tuple[float, float]] = {}
        for name, extrema in self.ranges.items():
            if name == "input.normalized":
                resolved[name] = self._include_zero(*extrema)
                continue
            if self.method in {"naive", "minmax"}:
                bounds = extrema
            elif self.method == "mean":
                sum_min, sum_max, count = self._range_sums[name]
                bounds = (sum_min / count, sum_max / count)
            else:
                values = self._values(name)
                if self.method == "entropy":
                    threshold = self._entropy_abs_threshold(values)
                    bounds = (-threshold, threshold)
                elif self.method == "coverage":
                    bounds = self._shortest_coverage_interval(values, self.coverage)
                elif self.method == "percentile":
                    tail = (100.0 - self.percentile) * 0.5
                    bounds = tuple(float(v) for v in np.percentile(values, (tail, 100.0 - tail)))
                elif self.method == "mse":
                    bounds = self._mse_range(values)
                else:
                    raise AssertionError(self.method)
            resolved[name] = self._include_zero(*bounds)
        return resolved

    def to_json(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "min": qmin,
                "max": qmax,
                "range_method": self.method,
                "exclude_register_tokens": self.exclude_register_tokens,
                "register_range_policy": self.register_range_policy,
            }
            for name, (qmin, qmax) in sorted(self.resolved_ranges().items())
        }


class TorchViTOpGraph(torch.nn.Module):
    def __init__(self, config: ViTOpConfig, weights: dict[str, np.ndarray], device: torch.device):
        super().__init__()
        self.config = config
        self.weights = {k: torch.from_numpy(v).to(device) for k, v in weights.items()}
        self.collector = RangeCollector()
        self.register_residual_scale = 1.0
        self.stable_attention_window = 0.0

    def _w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def _record(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return self.collector.record(name, tensor)

    def _token_groups(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        cls_end = 1 if cfg.has_cls_token else 0
        register_end = cls_end + cfg.num_register_tokens
        groups: dict[str, torch.Tensor] = {}
        if cls_end:
            groups["cls"] = tensor[:, :cls_end]
        if cfg.num_register_tokens:
            groups["registers"] = tensor[:, cls_end:register_end]
        groups["patches"] = tensor[:, register_end:]
        return groups

    def _record_token_groups(
        self, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        for group_name, group in self._token_groups(tensor).items():
            self._record(f"{name}.{group_name}", group)
        return tensor

    def _record_attention_heads(
        self, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        for head_id in range(self.config.num_attention_heads):
            self._record(
                f"{name}.h{head_id:02d}",
                tensor[:, head_id : head_id + 1],
            )
        return tensor

    def _scale_register_update(
        self, tensor: torch.Tensor, name: str
    ) -> torch.Tensor:
        scale = float(self.register_residual_scale)
        if scale == 1.0 or self.config.num_register_tokens == 0:
            return tensor
        groups = self._token_groups(tensor)
        groups["registers"] = groups["registers"] * scale
        return self._record(name, torch.cat(tuple(groups.values()), dim=1))

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
        qk_matmul = torch.matmul(q, k.transpose(-1, -2))
        qk_matmul = self._record(f"{c}.qk_matmul", qk_matmul)
        self._record_attention_heads(f"{c}.qk_matmul", qk_matmul)
        scores = qk_matmul * (1.0 / math.sqrt(cfg.head_dim))
        scores = self._record(f"{c}.attn_scores.raw", scores)
        self._record_attention_heads(f"{c}.attn_scores.raw", scores)
        if float(self.stable_attention_window) > 0.0:
            rowmax = scores.amax(dim=-1, keepdim=True)
            rowmax = self._record(f"{c}.attn_rowmax", rowmax)
            self._record_attention_heads(f"{c}.attn_rowmax", rowmax)
            scores = scores - rowmax
            scores = self._record(f"{c}.attn_centered", scores)
            self._record_attention_heads(f"{c}.attn_centered", scores)
            scores = torch.clamp(
                scores,
                min=-float(self.stable_attention_window),
                max=0.0,
            )
        scores = self._record(f"{c}.attn_scores", scores)
        self._record_attention_heads(f"{c}.attn_scores", scores)
        probs = torch.softmax(scores, dim=-1)
        probs = self._record(f"{c}.attn_probs", probs)
        self._record_attention_heads(f"{c}.attn_probs", probs)
        attn = torch.matmul(probs, v)
        attn = self._record(f"{c}.av_matmul", attn)
        self._record_attention_heads(f"{c}.av_matmul", attn)
        attn = attn.transpose(1, 2).reshape(bsz, seq_len, cfg.hidden_size)
        attn = self._record(f"{c}.attn", attn)
        proj = self._pointwise(attn, f"{c}.proj", cfg.hidden_size, f"{c}.proj")
        proj = self._scale_register_update(proj, f"{c}.proj.scaled")
        hidden = self._record(f"{c}.resid1", hidden + proj)
        normed2 = F.layer_norm(hidden, (cfg.hidden_size,), self._w(f"{c}.norm2.weight"), self._w(f"{c}.norm2.bias"), cfg.layer_norm_eps)
        normed2 = self._record(f"{c}.norm2", normed2)
        if cfg.use_gated_mlp:
            gate = self._pointwise(normed2, f"{c}.gate", cfg.intermediate_size, f"{c}.gate")
            gate = self._record(f"{c}.gate_act", F.silu(gate))
            up = self._pointwise(normed2, f"{c}.up", cfg.intermediate_size, f"{c}.up")
            mlp = self._record(f"{c}.mlp_mid", gate * up)
        else:
            mlp = self._pointwise(
                normed2, f"{c}.fc1", cfg.intermediate_size, f"{c}.fc1"
            )
            self._record_token_groups(f"{c}.fc1", mlp)
            mlp = F.gelu(mlp)
            mlp = self._record(f"{c}.mlp_mid", mlp)
            self._record_token_groups(f"{c}.mlp_mid", mlp)
        mlp = self._pointwise(mlp, f"{c}.fc2", cfg.hidden_size, f"{c}.fc2")
        self._record_token_groups(f"{c}.fc2.raw", mlp)
        mlp = self._scale_register_update(mlp, f"{c}.fc2.scaled")
        self._record_token_groups(f"{c}.fc2", mlp)
        return self._record(f"{c}.resid2", hidden + mlp)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        pixel_values = self._record("input.normalized", pixel_values)
        hidden = F.conv2d(pixel_values, self._w("patch.weight"), self._w("patch.bias"), stride=cfg.patch_size)
        hidden = self._record("patch.conv", hidden)
        hidden = hidden.reshape(pixel_values.shape[0], cfg.hidden_size, cfg.num_patches).transpose(1, 2)
        if cfg.prefix_len:
            prefix = self._w("prefix_tokens").expand(pixel_values.shape[0], -1, -1)
            hidden = torch.cat((prefix, hidden), dim=1)
            hidden = self._record("prefix_concat", hidden)
        if cfg.use_position_embeddings:
            hidden = hidden + self._w("pos_embed")
            hidden = self._record("add_position_embeddings", hidden)
        hidden = self._scale_register_update(hidden, "tokens.scaled")
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


def _normalized_uint8_input_range(config: ViTOpConfig) -> tuple[float, float]:
    """Return the complete normalized range produced by raw UINT8 image input."""
    mean = np.asarray(config.image_mean, dtype=np.float32)
    std = np.asarray(config.image_std, dtype=np.float32)
    if np.any(std <= 0.0):
        raise ValueError(f"image_std must be positive, got {config.image_std}")
    raw_bounds = np.asarray((0.0, 255.0), dtype=np.float32)[:, None]
    normalized = (
        raw_bounds * float(config.image_rescale_factor) - mean[None, :]
    ) / std[None, :]
    return float(normalized.min()), float(normalized.max())


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
    range_method: str = "minmax",
    range_percentile: float = 99.99,
    range_coverage: float = 99.99,
    max_range_samples: int = 65536,
    exclude_register_tokens: bool = False,
    register_range_policy: str = "include",
    register_residual_scale: float = 1.0,
    stable_attention_window: float = 0.0,
) -> dict[str, dict[str, Any]]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TorchViTOpGraph(config, weights, device).eval()
    model.register_residual_scale = float(register_residual_scale)
    model.stable_attention_window = float(stable_attention_window)
    if exclude_register_tokens:
        if register_range_policy != "include":
            raise ValueError(
                "--exclude-register-tokens-from-ranges cannot be combined "
                "with --register-range-policy"
            )
        register_range_policy = "exclude"
    model.collector = RangeCollector(
        range_method,
        num_observations=num_samples,
        max_samples_per_tensor=max_range_samples,
        percentile=range_percentile,
        coverage=range_coverage,
        token_axis_size=(
            config.seq_len if register_range_policy != "include" else None
        ),
        register_start=1 if config.has_cls_token else 0,
        num_register_tokens=(
            config.num_register_tokens
            if register_range_policy != "include"
            else 0
        ),
        register_range_policy=register_range_policy,
    )
    # Placeholder2 accepts raw UINT8 image bytes and applies rescale/mean/std
    # during the first forward. Seed the collector with the full theoretical
    # normalized range so input quantization does not depend on calibration
    # images containing exact black/white pixels.
    model.collector.set_fixed_range("input.normalized", *_normalized_uint8_input_range(config))
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


def compare_tfdl_quant_fp(
    config: ViTOpConfig,
    model_path: str | Path,
    quant_fb: str | Path,
    *,
    output_json: str | Path | None = None,
    calib_dir: str | Path | None = None,
    num_samples: int = 1,
    device_name: str | None = None,
    addon_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a dumped quant model with FP reference CLS/Patch tokens.

    The executor receives resized raw UINT8 NCHW. Placeholder2 performs the
    configured rescale/mean/std preprocessing on its first forward.
    """
    from TFDL2 import TFContext, TFExecutor

    quant_fb = Path(quant_fb).resolve()
    if not quant_fb.exists():
        raise FileNotFoundError(f"quant model does not exist: {quant_fb}")
    if addon_path is not None:
        _load_addon_if_needed(config, addon_path)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    result: dict[str, Any] = {
        "device": str(device),
        "quant_fb": str(quant_fb),
        "input_contract": "resized_raw_uint8_nchw",
        "frugal_mode": True,
        "attn_softmax_impl": True,
        "prefix_len": config.prefix_len,
        "samples": [],
    }
    try:
        reference = _load_reference_model(model_path, device)
        context = TFContext(path=str(quant_fb))
        executor = TFExecutor(
            context,
            {
                "UseHardware": False,
                "FrugalMode": True,
                "optimize": {"AttnSoftmaxImpl": True},
            },
        )
        inputs = executor.GetInputs()
        if len(inputs) != 1:
            raise RuntimeError(f"expected one TFDL input, got {len(inputs)}")
        samples = _load_compare_samples(
            config, calib_dir, max(1, num_samples), device
        )
        cls_cosines: list[float] = []
        patch_cosines: list[float] = []
        with torch.no_grad():
            for sample_id, (_, raw_nchw) in enumerate(samples):
                if raw_nchw.shape[0] != 1:
                    raise ValueError(
                        "--compare-tfdl-quant-fp currently expects batch=1"
                    )
                raw_uint8 = np.ascontiguousarray(
                    np.clip(np.rint(raw_nchw), 0, 255).astype(np.uint8)
                )
                # Compare against the exact normalized values produced from
                # the same UINT8 bytes consumed by the TFDL Placeholder.
                normalized = torch.from_numpy(
                    _preprocess_raw_nchw(config, raw_uint8)
                ).to(device)
                ref_tokens = (
                    _run_reference_model(reference, normalized.float())
                    .detach()
                    .cpu()
                    .numpy()
                )
                ref_cls = ref_tokens[:, 0]
                ref_patch = ref_tokens[:, config.prefix_len :]

                inputs[0].fromNumpy(raw_uint8)
                outputs = executor()
                if len(outputs) < 2:
                    raise RuntimeError(
                        f"expected CLS and token outputs, got {len(outputs)}"
                    )
                got_cls = outputs[0].toNumpy().astype(np.float32)
                got_tokens = outputs[1].toNumpy().astype(np.float32)
                got_patch = got_tokens[:, config.prefix_len :]
                stats = {
                    "cls": _compare_np(
                        "tfdl_quant.cls", ref_cls, got_cls
                    ),
                    "patch": _compare_np(
                        "tfdl_quant.patch", ref_patch, got_patch
                    ),
                }
                cls_cosines.append(stats["cls"]["cos"])
                patch_cosines.append(stats["patch"]["cos"])
                result["samples"].append(
                    {"sample": sample_id, "tfdl_quant": stats}
                )
                print(
                    f"[COMPARE-QUANT] sample={sample_id} "
                    f"cls_cos={stats['cls']['cos']:.8f} "
                    f"patch_cos={stats['patch']['cos']:.8f} "
                    f"cls_max_abs={stats['cls']['max_abs']:.6g} "
                    f"patch_max_abs={stats['patch']['max_abs']:.6g}"
                )
        result["mean"] = {
            "cls_cos": float(np.mean(cls_cosines)),
            "patch_cos": float(np.mean(patch_cosines)),
        }
        print(
            f"[COMPARE-QUANT-MEAN] samples={len(cls_cosines)} "
            f"cls_cos={result['mean']['cls_cos']:.8f} "
            f"patch_cos={result['mean']['patch_cos']:.8f}"
        )
        if output_json:
            Path(output_json).write_text(
                json.dumps(result, indent=2, sort_keys=True)
            )
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


def _token_group_specs(config: ViTOpConfig) -> tuple[tuple[str, int], ...]:
    groups: list[tuple[str, int]] = []
    if config.has_cls_token:
        groups.append(("cls", 1))
    if config.num_register_tokens:
        groups.append(("registers", config.num_register_tokens))
    groups.append(("patches", config.num_patches))
    return tuple(groups)


def _split_token_groups(hidden, config: ViTOpConfig):
    from TFDL2 import Op

    specs = _token_group_specs(config)
    return specs, tuple(
        Op.Slice(hidden, axis=1, split=tuple(size for _, size in specs))
    )


def _pointwise_token_group_op(
    ctx,
    hidden,
    config: ViTOpConfig,
    weight_name: str,
    bias_name: str,
    out_channels: int,
    symbol_map: dict[str, str],
    tag: str,
) -> dict[str, Any]:
    from TFDL2 import Op

    specs, groups = _split_token_groups(hidden, config)
    outputs: dict[str, Any] = {}
    for (group_name, token_count), group in zip(specs, groups):
        channels = int(
            ctx.GetParamSymbol(weight_name).shape[1]
            if hasattr(ctx.GetParamSymbol(weight_name), "shape")
            else config.hidden_size
        )
        group_4d = Op.Reshape(
            Op.Transpose(group, (0, 2, 1)),
            (1, channels, 1, token_count),
        )
        out_4d = _pointwise_4d_op(
            ctx,
            group_4d,
            weight_name,
            bias_name,
            out_channels,
            symbol_map,
            f"{tag}.{group_name}",
        )
        outputs[group_name] = Op.Transpose(
            Op.Reshape(out_4d, (1, out_channels, token_count)),
            (0, 2, 1),
        )
    return outputs


def _scale_register_update_op(
    hidden,
    config: ViTOpConfig,
    scale: float,
    symbol_map: dict[str, str],
    tag: str,
):
    from TFDL2 import Op

    if float(scale) == 1.0 or config.num_register_tokens == 0:
        return hidden
    specs, groups = _split_token_groups(hidden, config)
    scaled = []
    for (group_name, _), group in zip(specs, groups):
        if group_name == "registers":
            group = Op.Mul(group, float(scale))
        scaled.append(group)
    return _mark(symbol_map, tag, Op.Concat(tuple(scaled), axis=1))


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


def _build_block_op(
    ctx,
    hidden,
    config: ViTOpConfig,
    layer_id: int,
    symbol_map: dict[str, str],
    rope_sin=None,
    rope_cos=None,
    register_residual_scale: float = 1.0,
    split_mlp_token_groups: bool = False,
    split_attention_heads: bool = False,
    stable_attention_window: float = 0.0,
):
    from TFDL2 import Op

    c = f"layers.{layer_id}"
    normed = Op.LayerNorm2(hidden, ctx.GetParamSymbol(f"{c}.norm1.weight"), ctx.GetParamSymbol(f"{c}.norm1.bias"), axis=-1)
    normed = _mark(symbol_map, f"{c}.norm1", normed)
    norm1_transposed = _mark(
        symbol_map,
        f"{c}.norm1_transposed",
        Op.Transpose(normed, (0, 2, 1)),
    )
    normed_4d = _mark(
        symbol_map,
        f"{c}.norm1_conv_input",
        Op.Reshape(
            norm1_transposed,
            (1, config.hidden_size, *config.seq_map_hw),
        ),
    )
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
    _mark(symbol_map, f"{c}.q_matmul_input", q3)
    _mark(symbol_map, f"{c}.k_matmul_input", k3)
    _mark(symbol_map, f"{c}.v_matmul_input", v3)
    if split_attention_heads:
        head_splits = (1,) * config.num_attention_heads
        q_heads = Op.Slice(q3, axis=0, split=head_splits)
        k_heads = Op.Slice(k3, axis=0, split=head_splits)
        v_heads = Op.Slice(v3, axis=0, split=head_splits)
        attn_heads = []
        for head_id, (q_head, k_head, v_head) in enumerate(
            zip(q_heads, k_heads, v_heads)
        ):
            h = f"h{head_id:02d}"
            qk_head = _mark(
                symbol_map,
                f"{c}.qk_matmul.{h}",
                Op.MatMul(q_head, k_head, transA=False, transB=False),
            )
            scores_head = _mark(
                symbol_map,
                f"{c}.attn_scores.raw.{h}",
                Op.Mul(qk_head, 1.0 / math.sqrt(config.head_dim)),
            )
            if float(stable_attention_window) > 0.0:
                rowmax = _mark(
                    symbol_map,
                    f"{c}.attn_rowmax.{h}",
                    Op.ReduceMax(scores_head, dims=(2,), keep_dims=True),
                )
                scores_head = _mark(
                    symbol_map,
                    f"{c}.attn_centered.{h}",
                    Op.Sub(scores_head, rowmax),
                )
                window = float(stable_attention_window)
                scores_head = Op.Sub(
                    Op.ReLU(Op.Add(scores_head, window)),
                    window,
                )
            scores_head = _mark(
                symbol_map,
                f"{c}.attn_scores.{h}",
                scores_head,
            )
            probs_head = _mark(
                symbol_map,
                f"{c}.attn_probs.{h}",
                Op.Softmax(scores_head, axis=2),
            )
            attn_heads.append(
                _mark(
                    symbol_map,
                    f"{c}.av_matmul.{h}",
                    Op.MatMul(probs_head, v_head, transA=False, transB=False),
                )
            )
        attn = _mark(
            symbol_map,
            f"{c}.av_matmul",
            Op.Concat(tuple(attn_heads), axis=0),
        )
    else:
        qk_matmul = _mark(
            symbol_map,
            f"{c}.qk_matmul",
            Op.MatMul(q3, k3, transA=False, transB=False),
        )
        scores = _mark(
            symbol_map,
            f"{c}.attn_scores",
            Op.Mul(qk_matmul, 1.0 / math.sqrt(config.head_dim)),
        )
        probs = _mark(symbol_map, f"{c}.attn_probs", Op.Softmax(scores, axis=2))
        attn = _mark(
            symbol_map,
            f"{c}.av_matmul",
            Op.MatMul(probs, v3, transA=False, transB=False),
        )
    attn_4d = _mark(symbol_map, f"{c}.attn", _attention_to_conv1d(attn, config))
    proj = _mark(
        symbol_map,
        f"{c}.proj_tokens",
        _conv1d_to_tokens(
        _pointwise_4d_op(ctx, attn_4d, f"{c}.proj.weight", f"{c}.proj.bias", config.hidden_size, symbol_map, f"{c}.proj"),
        config.hidden_size,
        config,
        ),
    )
    proj = _scale_register_update_op(
        proj,
        config,
        register_residual_scale,
        symbol_map,
        f"{c}.proj.scaled",
    )
    hidden = _mark(symbol_map, f"{c}.resid1", Op.Add(hidden, proj))
    normed2 = _mark(symbol_map, f"{c}.norm2", Op.LayerNorm2(hidden, ctx.GetParamSymbol(f"{c}.norm2.weight"), ctx.GetParamSymbol(f"{c}.norm2.bias"), axis=-1))
    norm2_transposed = _mark(
        symbol_map,
        f"{c}.norm2_transposed",
        Op.Transpose(normed2, (0, 2, 1)),
    )
    normed2_4d = _mark(
        symbol_map,
        f"{c}.norm2_conv_input",
        Op.Reshape(
            norm2_transposed,
            (1, config.hidden_size, *config.seq_map_hw),
        ),
    )
    if config.use_gated_mlp:
        gate = _pointwise_4d_op(ctx, normed2_4d, f"{c}.gate.weight", f"{c}.gate.bias", config.intermediate_size, symbol_map, f"{c}.gate")
        gate = _mark(symbol_map, f"{c}.gate_act", Op.Swish(gate))
        up = _pointwise_4d_op(ctx, normed2_4d, f"{c}.up.weight", f"{c}.up.bias", config.intermediate_size, symbol_map, f"{c}.up")
        mlp = _mark(symbol_map, f"{c}.mlp_mid", Op.Mul(gate, up))
    else:
        if split_mlp_token_groups:
            fc1_groups = _pointwise_token_group_op(
                ctx,
                normed2,
                config,
                f"{c}.fc1.weight",
                f"{c}.fc1.bias",
                config.intermediate_size,
                symbol_map,
                f"{c}.fc1",
            )
            fc2_groups = []
            for group_name, fc1_group in fc1_groups.items():
                mid = _mark(
                    symbol_map,
                    f"{c}.mlp_mid.{group_name}",
                    Op.GeLU(fc1_group),
                )
                token_count = dict(_token_group_specs(config))[group_name]
                mid_4d = Op.Reshape(
                    Op.Transpose(mid, (0, 2, 1)),
                    (1, config.intermediate_size, 1, token_count),
                )
                raw = _pointwise_4d_op(
                    ctx,
                    mid_4d,
                    f"{c}.fc2.weight",
                    f"{c}.fc2.bias",
                    config.hidden_size,
                    symbol_map,
                    f"{c}.fc2.raw.{group_name}",
                )
                out = Op.Transpose(
                    Op.Reshape(raw, (1, config.hidden_size, token_count)),
                    (0, 2, 1),
                )
                if group_name == "registers" and float(register_residual_scale) != 1.0:
                    out = Op.Mul(out, float(register_residual_scale))
                out = _mark(symbol_map, f"{c}.fc2.{group_name}", out)
                fc2_groups.append(out)
            mlp = _mark(
                symbol_map,
                f"{c}.fc2.scaled",
                Op.Concat(tuple(fc2_groups), axis=1),
            )
            symbol_map[f"{c}.fc2"] = str(mlp)
            return _mark(symbol_map, f"{c}.resid2", Op.Add(hidden, mlp))
        mlp = _pointwise_4d_op(ctx, normed2_4d, f"{c}.fc1.weight", f"{c}.fc1.bias", config.intermediate_size, symbol_map, f"{c}.fc1")
        mlp = _mark(symbol_map, f"{c}.mlp_mid", Op.GeLU(mlp))
    mlp = _mark(
        symbol_map,
        f"{c}.fc2_tokens",
        _conv1d_to_tokens(
        _pointwise_4d_op(ctx, mlp, f"{c}.fc2.weight", f"{c}.fc2.bias", config.hidden_size, symbol_map, f"{c}.fc2"),
        config.hidden_size,
        config,
        ),
    )
    return _mark(symbol_map, f"{c}.resid2", Op.Add(hidden, mlp))


def build_vit_tfdl_graph(
    config: ViTOpConfig,
    weights: dict[str, np.ndarray],
    range_json: str | Path | None = None,
    create_executor: bool = False,
    register_residual_scale: float = 1.0,
    split_mlp_token_groups: bool = False,
    split_attention_heads: bool = False,
    stable_attention_window: float = 0.0,
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
        pixel_values = _mark(symbol_map, "input.normalized", pixel_values)
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
        hidden = _mark(
            symbol_map,
            "patch.tokens",
            Op.Transpose(
                Op.Reshape(
                    hidden,
                    (1, config.hidden_size, config.num_patches),
                ),
                (0, 2, 1),
            ),
        )
        if config.prefix_len:
            hidden = _mark(
                symbol_map,
                "prefix_concat",
                Op.Concat(
                    (ctx.GetParamSymbol("prefix_tokens"), hidden),
                    axis=1,
                ),
            )
        if config.use_position_embeddings:
            hidden = Op.Add(hidden, ctx.GetParamSymbol("pos_embed"))
            hidden = _mark(symbol_map, "add_position_embeddings", hidden)
        hidden = _scale_register_update_op(
            hidden,
            config,
            register_residual_scale,
            symbol_map,
            "tokens.scaled",
        )
        hidden = _mark(symbol_map, "tokens", hidden)
        rope_sin = ctx.GetParamSymbol("rope_sin") if config.use_rope else None
        rope_cos = ctx.GetParamSymbol("rope_cos") if config.use_rope else None
        for layer_id in range(config.num_hidden_layers):
            hidden = _build_block_op(
                ctx,
                hidden,
                config,
                layer_id,
                symbol_map,
                rope_sin,
                rope_cos,
                register_residual_scale=register_residual_scale,
                split_mlp_token_groups=split_mlp_token_groups,
                split_attention_heads=split_attention_heads,
                stable_attention_window=stable_attention_window,
            )
        hidden = _mark(symbol_map, "final_norm", Op.LayerNorm2(hidden, ctx.GetParamSymbol("norm.weight"), ctx.GetParamSymbol("norm.bias"), axis=-1))
        cls = Op.Gather2(hidden, 0, 1)
        cls = _mark(symbol_map, "output.cls", Op.Reshape(cls, (1, config.hidden_size)))
        output_names = [str(cls), str(hidden)]
    ctx.SetOutputs(output_names)

    if range_json:
        ranges = _load_range_json(range_json)
        input_tag = "input.normalized"
        if input_tag not in ranges:
            raise ValueError(
                f"range JSON is missing required {input_tag!r}; regenerate it "
                "before quantization so Placeholder input scale/zero-point are valid"
            )
        input_min, input_max = ranges[input_tag]
        if not np.isfinite(input_min) or not np.isfinite(input_max) or input_min >= input_max:
            raise ValueError(
                f"invalid {input_tag!r} range: min={input_min}, max={input_max}"
            )
        for tag, actual in symbol_map.items():
            if tag in ranges:
                qmin, qmax = ranges[tag]
                if not ctx.AddInt8Config(actual, float(qmax), float(qmin)):
                    raise RuntimeError(f"failed to add int8 config for {tag} -> {actual}")

    executor = None
    if create_executor:
        executor = TFExecutor(
            ctx,
            {
                "UseHardware": False,
                "FrugalMode": True,
                "optimize": {"AttnSoftmaxImpl": True},
            },
        )
    return ctx, executor, [input_name], output_names, symbol_map


def dump_context(ctx: Any, output: str | Path) -> None:
    output = str(output)
    if output.endswith(".fb"):
        output = output[:-3]
    ctx.Dump(output)


def quantize_with_ranges(
    ctx: Any,
    input_names: list[str],
    output_names: list[str],
    output: str | Path | None,
    *,
    extra_stopquanttensors: tuple[str, ...] = (),
    avoidtensors: tuple[str, ...] = (),
) -> None:
    from TFDL2 import TFCalibration, CalibrationMode
    from TFDL2.Common import TFDataType

    calib = TFCalibration(
        ctx,
        CalibrationMode.Naive,
        {
            "UseHardware": False,
            "FrugalMode": True,
            "optimize": {"AttnSoftmaxImpl": True},
        },
    )
    calib.Quantize(
        {name: TFDataType.TFDL_UINT8 for name in input_names},
        stopquanttensors=tuple(output_names) + tuple(extra_stopquanttensors),
        avoidtensors=tuple(avoidtensors),
        MergeConcate=False,
        Perchannel=True,
    )
    ctx.SetOutputs(output_names)
    if output is not None:
        dump_context(ctx, output)


def _range_item(
    ranges: dict[str, Any],
    name: str,
) -> tuple[float, float]:
    if name not in ranges:
        raise KeyError(f"range JSON is missing {name!r}")
    value = ranges[name]
    if isinstance(value, dict):
        qmin, qmax = float(value["min"]), float(value["max"])
    else:
        qmin, qmax = float(value[0]), float(value[1])
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmin >= qmax:
        raise ValueError(f"invalid range for {name}: min={qmin}, max={qmax}")
    return qmin, qmax


def _modify_range_strings(
    ranges: dict[str, Any],
    name: str,
) -> tuple[list[str], list[str]]:
    qmin, qmax = _range_item(ranges, name)
    # Modify's JSON parser expects strings here. Numeric arrays may parse but
    # have historically produced a zero qscale for inserted Quantize nodes.
    return [repr(qmin)], [repr(qmax)]


def select_outlier_branches(
    config: ViTOpConfig,
    range_json: str | Path,
    top_k: int,
) -> dict[str, Any]:
    """Rank residual-merge Attention/MLP branches by absolute range."""
    if top_k < 0:
        raise ValueError("--outlier-bypass-top-k must be non-negative")
    ranges = json.loads(Path(range_json).read_text())
    candidates: list[dict[str, Any]] = []
    for layer_id in range(config.num_hidden_layers):
        for kind, suffix in (("attn", "proj"), ("mlp", "fc2")):
            tag = f"layers.{layer_id}.{suffix}"
            qmin, qmax = _range_item(ranges, tag)
            candidates.append(
                {
                    "kind": kind,
                    "layer": layer_id,
                    "tag": tag,
                    "min": qmin,
                    "max": qmax,
                    "abs_max": max(abs(qmin), abs(qmax)),
                }
            )
    candidates.sort(
        key=lambda item: (-item["abs_max"], item["layer"], item["kind"])
    )
    for rank, item in enumerate(candidates, 1):
        item["rank"] = rank
    selected = candidates[: min(top_k, len(candidates))]
    return {
        "top_k": int(top_k),
        "candidate_count": len(candidates),
        "selected": selected,
        "fp_attn_layers": [
            item["layer"] for item in selected if item["kind"] == "attn"
        ],
        "fp_mlp_layers": [
            item["layer"] for item in selected if item["kind"] == "mlp"
        ],
        "ranking": candidates,
    }


def apply_outlier_bypass_modify(
    ctx: Any,
    config: ViTOpConfig,
    weights: dict[str, np.ndarray],
    range_json: str | Path,
    symbol_map: dict[str, str],
    *,
    top_k: int,
    fp16_export: bool = False,
    dump_modify_json: str | Path | None = None,
    dump_bypass_report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Post-quant Modify: INT8 branches, floating residuals and Top-K bypass.

    The quantizer runs before this function. Selected Attention candidates
    bypass only the output projection; selected MLP candidates bypass the
    complete MLP. This preserves INT8 QKV/Softmax/AV work on the NPU.
    """
    if config.prefix_len <= 0:
        raise ValueError("mixed Attention/MLP profile requires prefix tokens")

    ranges = json.loads(Path(range_json).read_text())
    report = select_outlier_branches(config, range_json, top_k)
    fp_attn_layers = frozenset(report["fp_attn_layers"])
    fp_mlp_layers = frozenset(report["fp_mlp_layers"])
    branch_dtype = "TFDtypeFp16" if fp16_export else "TFDtypeFp32"
    param_dtype = np.float16 if fp16_export else np.float32
    dequant_options: dict[str, Any] = (
        {"param": {"dstType": "TFDtypeFp16"}} if fp16_export else {}
    )
    float_param_names: set[str] = {"prefix_tokens"}
    layers: list[dict[str, Any]] = []

    def symbol(tag: str) -> str:
        if tag not in symbol_map:
            raise KeyError(
                f"symbol map is missing {tag!r}; this profile is incompatible "
                "with the selected graph transformation options"
            )
        return symbol_map[tag]

    patch_tokens = symbol("patch.tokens")
    prefix_concat = symbol("prefix_concat")
    dequant_patch = "PostDeQuant_PatchTokens"
    layers.extend(
        (
            {
                "input": [patch_tokens],
                "layerName": dequant_patch,
                "layerType": "DeQuantize",
                "output": [dequant_patch],
                **dequant_options,
            },
            {
                "input": ["PostFP.prefix_tokens", dequant_patch],
                "layerName": prefix_concat,
                "outputDataType": branch_dtype,
            },
        )
    )

    first_hidden = prefix_concat
    tokens = symbol("tokens")
    if config.use_position_embeddings:
        float_param_names.add("pos_embed")
        layers.append(
            {
                "input": [prefix_concat, "PostFP.pos_embed"],
                "layerName": tokens,
                "outputDataType": branch_dtype,
            }
        )
        first_hidden = tokens
    elif tokens != prefix_concat:
        raise ValueError(
            "Top-K bypass currently requires --register-residual-scale=1 for "
            "DINOv3 so the prefix concat is the first block input"
        )

    for layer_id in range(config.num_hidden_layers):
        prefix = f"layers.{layer_id}"
        hidden = (
            first_hidden
            if layer_id == 0
            else symbol(f"layers.{layer_id - 1}.resid2")
        )
        norm1 = symbol(f"{prefix}.norm1")
        norm1_transposed = symbol(f"{prefix}.norm1_transposed")
        quant_norm1 = f"PostQuant_AttnInput_{layer_id}"
        norm1_min, norm1_max = _modify_range_strings(
            ranges, f"{prefix}.norm1"
        )
        attn = symbol(f"{prefix}.attn")
        proj = symbol(f"{prefix}.proj")
        proj_tokens = symbol(f"{prefix}.proj_tokens")
        resid1 = symbol(f"{prefix}.resid1")
        norm2 = symbol(f"{prefix}.norm2")
        norm2_transposed = symbol(f"{prefix}.norm2_transposed")
        norm2_conv_input = symbol(f"{prefix}.norm2_conv_input")
        quant_norm2 = f"PostQuant_MLPInput_{layer_id}"
        norm2_min, norm2_max = _modify_range_strings(
            ranges, f"{prefix}.norm2"
        )
        fc2 = symbol(f"{prefix}.fc2")
        fc2_tokens = symbol(f"{prefix}.fc2_tokens")
        resid2 = symbol(f"{prefix}.resid2")

        layers.extend(
            (
                {
                    "input": [
                        hidden,
                        f"{prefix}.norm1.weight",
                        f"{prefix}.norm1.bias",
                    ],
                    "layerName": norm1,
                    "outputDataType": "TFDtypeFp32",
                },
                {
                    "input": [norm1],
                    "layerName": quant_norm1,
                    "layerType": "Quantize",
                    "output": [quant_norm1],
                    "outputDataType": "TFDtypeUint8",
                    "OutDataMin": norm1_min,
                    "OutDataMax": norm1_max,
                },
                {
                    "input": [quant_norm1],
                    "layerName": norm1_transposed,
                    "outputDataType": "TFDtypeUint8",
                },
            )
        )

        if layer_id in fp_attn_layers:
            dequant_attn = f"PostDeQuant_AttnCore_{layer_id}"
            float_param_names.update(
                (f"{prefix}.proj.weight", f"{prefix}.proj.bias")
            )
            layers.extend(
                (
                    {
                        "input": [attn],
                        "layerName": dequant_attn,
                        "layerType": "DeQuantize",
                        "output": [dequant_attn],
                        **dequant_options,
                    },
                    {
                        "input": [
                            dequant_attn,
                            f"PostFP.{prefix}.proj.weight",
                            f"PostFP.{prefix}.proj.bias",
                        ],
                        "layerName": proj,
                        "outputDataType": branch_dtype,
                    },
                    {
                        "layerName": proj_tokens,
                        "outputDataType": branch_dtype,
                    },
                )
            )
            attn_residual_input = proj_tokens
        else:
            dequant_proj = f"PostDeQuant_AttnOutput_{layer_id}"
            layers.append(
                {
                    "input": [proj_tokens],
                    "layerName": dequant_proj,
                    "layerType": "DeQuantize",
                    "output": [dequant_proj],
                    **dequant_options,
                }
            )
            attn_residual_input = dequant_proj

        layers.extend(
            (
                {
                    "input": [hidden, attn_residual_input],
                    "layerName": resid1,
                    "outputDataType": branch_dtype,
                },
                {
                    "input": [
                        resid1,
                        f"{prefix}.norm2.weight",
                        f"{prefix}.norm2.bias",
                    ],
                    "layerName": norm2,
                    "outputDataType": "TFDtypeFp32",
                },
            )
        )

        if layer_id in fp_mlp_layers:
            # Keep LayerNorm computation in FP32. The Transpose output is the
            # explicit FP32->FP16 boundary when --fp16-export is enabled.
            layers.extend(
                (
                    {
                        "input": [norm2],
                        "layerName": norm2_transposed,
                        "outputDataType": branch_dtype,
                    },
                    {
                        "layerName": norm2_conv_input,
                        "outputDataType": branch_dtype,
                    },
                )
            )
            if config.use_gated_mlp:
                gate = symbol(f"{prefix}.gate")
                gate_act = symbol(f"{prefix}.gate_act")
                up = symbol(f"{prefix}.up")
                mlp_mid = symbol(f"{prefix}.mlp_mid")
                for stem in ("gate", "up", "fc2"):
                    float_param_names.update(
                        (
                            f"{prefix}.{stem}.weight",
                            f"{prefix}.{stem}.bias",
                        )
                    )
                layers.extend(
                    (
                        {
                            "input": [
                                norm2_conv_input,
                                f"PostFP.{prefix}.gate.weight",
                                f"PostFP.{prefix}.gate.bias",
                            ],
                            "layerName": gate,
                            "outputDataType": branch_dtype,
                        },
                        {
                            "input": [gate],
                            "layerName": gate_act,
                            "outputDataType": branch_dtype,
                        },
                        {
                            "input": [
                                norm2_conv_input,
                                f"PostFP.{prefix}.up.weight",
                                f"PostFP.{prefix}.up.bias",
                            ],
                            "layerName": up,
                            "outputDataType": branch_dtype,
                        },
                        {
                            "input": [gate_act, up],
                            "layerName": mlp_mid,
                            "outputDataType": branch_dtype,
                        },
                    )
                )
                mlp_input = mlp_mid
            else:
                fc1 = symbol(f"{prefix}.fc1")
                mlp_mid = symbol(f"{prefix}.mlp_mid")
                float_param_names.update(
                    (
                        f"{prefix}.fc1.weight",
                        f"{prefix}.fc1.bias",
                        f"{prefix}.fc2.weight",
                        f"{prefix}.fc2.bias",
                    )
                )
                layers.extend(
                    (
                        {
                            "input": [
                                norm2_conv_input,
                                f"PostFP.{prefix}.fc1.weight",
                                f"PostFP.{prefix}.fc1.bias",
                            ],
                            "layerName": fc1,
                            "outputDataType": branch_dtype,
                        },
                        {
                            "input": [fc1],
                            "layerName": mlp_mid,
                            "outputDataType": branch_dtype,
                        },
                    )
                )
                mlp_input = mlp_mid
            layers.extend(
                (
                    {
                        "input": [
                            mlp_input,
                            f"PostFP.{prefix}.fc2.weight",
                            f"PostFP.{prefix}.fc2.bias",
                        ],
                        "layerName": fc2,
                        "outputDataType": branch_dtype,
                    },
                    {
                        "layerName": fc2_tokens,
                        "outputDataType": branch_dtype,
                    },
                )
            )
            mlp_residual_input = fc2_tokens
        else:
            dequant_mlp = f"PostDeQuant_MLPOutput_{layer_id}"
            layers.extend(
                (
                    {
                        "input": [norm2],
                        "layerName": quant_norm2,
                        "layerType": "Quantize",
                        "output": [quant_norm2],
                        "outputDataType": "TFDtypeUint8",
                        "OutDataMin": norm2_min,
                        "OutDataMax": norm2_max,
                    },
                    {
                        "input": [quant_norm2],
                        "layerName": norm2_transposed,
                        "outputDataType": "TFDtypeUint8",
                    },
                    {
                        "input": [fc2_tokens],
                        "layerName": dequant_mlp,
                        "layerType": "DeQuantize",
                        "output": [dequant_mlp],
                        **dequant_options,
                    },
                )
            )
            mlp_residual_input = dequant_mlp

        layers.append(
            {
                "input": [resid1, mlp_residual_input],
                "layerName": resid2,
                "outputDataType": branch_dtype,
            }
        )

    final_norm = symbol("final_norm")
    final_hidden = symbol(
        f"layers.{config.num_hidden_layers - 1}.resid2"
    )
    layers.append(
        {
            "input": [final_hidden, "norm.weight", "norm.bias"],
            "layerName": final_norm,
            "outputDataType": "TFDtypeFp32",
        }
    )

    restored_params = {
        f"PostFP.{name}": np.ascontiguousarray(weights[name], dtype=param_dtype)
        for name in sorted(float_param_names)
    }
    ctx.RegisterParamToContext(**restored_params)
    modify = {"AddOnPass": [], "DeleteLayer": [], "Layer": layers}
    ctx.Modify(modify)

    report.update(
        {
            "fp16_export": bool(fp16_export),
            "restored_param_dtype": str(np.dtype(param_dtype)),
            "restored_params": sorted(restored_params),
            "modified_layer_entries": len(layers),
        }
    )
    if dump_modify_json:
        Path(dump_modify_json).write_text(
            json.dumps(modify, indent=2, sort_keys=True)
        )
    if dump_bypass_report_json:
        Path(dump_bypass_report_json).write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )
    return report


def build_arg_parser(default_arch: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and test ViT/DINOv2/DINOv3 TFDL2 graphs. Model depth, "
            "width and token layout are read from config.json."
        )
    )
    parser.add_argument("--arch", default=default_arch or "vit", choices=("vit", "dinov2", "dinov3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-size", type=int, nargs='+', default=None)
    parser.add_argument("--calib-dir", default=None)
    parser.add_argument("--num-calib", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default prefers CUDA.")
    parser.add_argument("--compare-reference", action="store_true", help="Compare PyTorch Op-equivalent graph with the original transformers model")
    parser.add_argument("--compare-tfdl-fp", action="store_true", help="Compare the built TFDL float executor with the original transformers model")
    parser.add_argument(
        "--compare-tfdl-quant-fp",
        nargs="?",
        const="__generated__",
        default=None,
        metavar="QUANT_FB",
        help=(
            "Compare quantized TFDL CLS/Patch tokens with the FP transformers "
            "reference. Omit QUANT_FB to test --dump-quant-fb."
        ),
    )
    parser.add_argument("--compare-json", default=None, help="Optional path to save cosine/max error comparison stats")
    parser.add_argument("--dump-minmax-json", default=None)
    parser.add_argument(
        "--range-method",
        default="minmax",
        choices=RangeCollector.METHODS,
        help="Range algorithm implemented in this script; SDK still receives only final min/max via AddInt8Config",
    )
    parser.add_argument(
        "--range-methods",
        nargs="+",
        choices=RangeCollector.METHODS,
        default=None,
        help=(
            "Run several script-side range methods in one command. Output "
            "paths may contain {method}; otherwise the method is inserted "
            "before .minmax.json/.quant.fb."
        ),
    )
    parser.add_argument(
        "--range-percentile",
        type=float,
        default=99.99,
        help="Central percentage retained by --range-method percentile",
    )
    parser.add_argument(
        "--range-coverage",
        type=float,
        default=99.99,
        help="Shortest interval percentage retained by --range-method coverage",
    )
    parser.add_argument(
        "--max-range-samples",
        type=int,
        default=65536,
        help="Maximum sampled activation values retained per tensor for clipping methods",
    )
    parser.add_argument(
        "--exclude-register-tokens-from-ranges",
        action="store_true",
        help="Exclude register-token positions from activation range statistics while retaining CLS and patch tokens",
    )
    parser.add_argument(
        "--register-range-policy",
        default="include",
        choices=("include", "exclude", "role-aware"),
        help=(
            "Register-token calibration policy. role-aware preserves register "
            "K/V and key columns while excluding register query/output outliers."
        ),
    )
    parser.add_argument("--range-json", default=None, help="Existing min/max JSON to map to TFDL AddInt8Config")
    parser.add_argument("--dump-fb", default=None)
    parser.add_argument("--dump-quant-fb", default=None)
    parser.add_argument("--dump-symbol-map", default=None)
    parser.add_argument(
        "--outlier-bypass-top-k",
        type=int,
        default=0,
        help=(
            "After quantization, globally rank Attention proj and MLP fc2 "
            "merge ranges and bypass INT8 for the largest K branches."
        ),
    )
    parser.add_argument(
        "--fp16-export",
        action="store_true",
        help=(
            "Use FP16 for post-quant DeQuant destinations, prefix parameters "
            "and restored Top-K branch parameters/outputs."
        ),
    )
    parser.add_argument(
        "--dump-modify-json",
        default=None,
        help="Optional path for the generated post-quant Modify JSON",
    )
    parser.add_argument(
        "--dump-bypass-report-json",
        default=None,
        help="Optional path for full branch range ranking and selected Top-K",
    )
    parser.add_argument(
        "--dequant-final-norm-input",
        action="store_true",
        help="Stop quantization after the last residual so SDK inserts a Dequant boundary before final LayerNorm",
    )
    parser.add_argument(
        "--register-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Exact pre-norm reparameterization scale for register residual "
            "states and their branch updates; try 0.25 or 0.125."
        ),
    )
    parser.add_argument(
        "--split-mlp-token-groups",
        action="store_true",
        help=(
            "Run CLS/register/patch MLP projections as separate per-channel "
            "INT8 Conv branches so each token role gets an independent scale."
        ),
    )
    parser.add_argument(
        "--split-attention-heads",
        action="store_true",
        help=(
            "Run each attention head's QK/Softmax/AV path separately so all "
            "activation MatMuls and Softmax tensors get per-head INT8 scales."
        ),
    )
    parser.add_argument(
        "--stable-attention-window",
        type=float,
        default=0.0,
        help=(
            "Subtract each attention row max and clamp the result to "
            "[-window, 0] before Softmax; 0 disables this exact/stable form."
        ),
    )
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


def _method_output_path(
    path: str | Path | None,
    method: str,
    multiple: bool,
) -> str | None:
    if path is None:
        return None
    value = str(path)
    if "{method}" in value:
        return value.format(method=method)
    if not multiple:
        return value
    for compound_suffix in (
        ".minmax.json",
        ".quant.fb",
        ".modify.json",
        ".report.json",
    ):
        if value.endswith(compound_suffix):
            return (
                value[: -len(compound_suffix)]
                + f".{method}"
                + compound_suffix
            )
    output = Path(value)
    return str(
        output.with_name(f"{output.stem}.{method}{output.suffix}")
    )


def main(argv: list[str] | None = None, default_arch: str | None = None) -> None:
    args = build_arg_parser(default_arch).parse_args(argv)
    if not (0.0 < float(args.register_residual_scale) <= 1.0):
        raise ValueError("--register-residual-scale must be in (0, 1]")
    if float(args.stable_attention_window) < 0.0:
        raise ValueError("--stable-attention-window must be non-negative")
    if args.stable_attention_window and not args.split_attention_heads:
        raise ValueError(
            "--stable-attention-window currently requires --split-attention-heads"
        )
    if args.outlier_bypass_top_k < 0:
        raise ValueError("--outlier-bypass-top-k must be non-negative")
    if (
        (args.outlier_bypass_top_k or args.fp16_export)
        and not args.dump_quant_fb
    ):
        raise ValueError(
            "--outlier-bypass-top-k/--fp16-export requires --dump-quant-fb"
        )
    if (
        (args.outlier_bypass_top_k or args.fp16_export)
        and args.split_mlp_token_groups
    ):
        raise ValueError(
            "Top-K post-quant bypass is incompatible with "
            "--split-mlp-token-groups"
        )
    range_methods = list(
        dict.fromkeys(args.range_methods or (args.range_method,))
    )
    multiple_methods = len(range_methods) > 1
    if multiple_methods and not args.dump_minmax_json:
        raise ValueError(
            "--range-methods with multiple methods requires "
            "--dump-minmax-json so each method has its own range file"
        )
    if multiple_methods and args.range_json:
        raise ValueError(
            "--range-json cannot be shared by several --range-methods"
        )
    if (
        args.compare_tfdl_quant_fp == "__generated__"
        and not args.dump_quant_fb
    ):
        raise ValueError(
            "--compare-tfdl-quant-fp without a path requires --dump-quant-fb"
        )

    config = ViTOpConfig.from_model_path(args.model_path, args.arch, image_size=args.image_size)
    raw = load_safetensors(args.model_path)
    weights = canonicalize_weights(raw, config)
    comparison_results: dict[str, Any] = {}
    if args.compare_reference or args.compare_tfdl_fp:
        comparison_results["fp_graph"] = compare_with_reference(
            config,
            weights,
            args.model_path,
            output_json=None,
            calib_dir=args.calib_dir,
            num_samples=args.num_calib,
            device_name=args.device,
            compare_torch_op=args.compare_reference,
            compare_tfdl_fp=args.compare_tfdl_fp,
            addon_path=args.addon_path,
        )

    needs_graph = bool(
        args.dump_fb
        or args.dump_quant_fb
        or args.dump_symbol_map
    )
    if needs_graph or args.compare_tfdl_quant_fp:
        _load_addon_if_needed(config, args.addon_path)

    generated_quant_paths: dict[str, str] = {}
    for method_index, method in enumerate(range_methods):
        range_json = args.range_json
        generated_range_json = _method_output_path(
            args.dump_minmax_json, method, multiple_methods
        )
        if generated_range_json:
            collect_minmax_json(
                config,
                weights,
                generated_range_json,
                calib_dir=args.calib_dir,
                num_samples=args.num_calib,
                device_name=args.device,
                range_method=method,
                range_percentile=args.range_percentile,
                range_coverage=args.range_coverage,
                max_range_samples=args.max_range_samples,
                exclude_register_tokens=args.exclude_register_tokens_from_ranges,
                register_range_policy=args.register_range_policy,
                register_residual_scale=args.register_residual_scale,
                stable_attention_window=args.stable_attention_window,
            )
            range_json = generated_range_json

        if not needs_graph:
            continue
        ctx, _, input_names, output_names, symbol_map = build_vit_tfdl_graph(
            config,
            weights,
            range_json=range_json,
            register_residual_scale=args.register_residual_scale,
            split_mlp_token_groups=args.split_mlp_token_groups,
            split_attention_heads=args.split_attention_heads,
            stable_attention_window=args.stable_attention_window,
        )
        if range_json:
            annotate_minmax_json_with_symbol_map(range_json, symbol_map)
        if args.dump_symbol_map and method_index == 0:
            Path(args.dump_symbol_map).write_text(
                json.dumps(symbol_map, indent=2, sort_keys=True)
            )
        if args.dump_fb and method_index == 0:
            dump_context(ctx, args.dump_fb)
        if args.dump_quant_fb:
            if not range_json:
                raise ValueError(
                    "--dump-quant-fb requires --range-json or "
                    "--dump-minmax-json"
                )
            extra_stopquanttensors = ()
            if args.dequant_final_norm_input:
                extra_stopquanttensors = (
                    symbol_map[f"layers.{config.num_hidden_layers - 1}.resid2"],
                )
            postquant_modify = bool(
                args.outlier_bypass_top_k or args.fp16_export
            )
            quantize_with_ranges(
                ctx,
                input_names,
                output_names,
                None if postquant_modify else _method_output_path(
                    args.dump_quant_fb, method, multiple_methods
                ),
                extra_stopquanttensors=extra_stopquanttensors,
            )
            quant_output = _method_output_path(
                args.dump_quant_fb, method, multiple_methods
            )
            if quant_output is None:
                raise AssertionError("quant output path was unexpectedly None")
            if postquant_modify:
                report = apply_outlier_bypass_modify(
                    ctx,
                    config,
                    weights,
                    range_json,
                    symbol_map,
                    top_k=args.outlier_bypass_top_k,
                    fp16_export=args.fp16_export,
                    dump_modify_json=_method_output_path(
                        args.dump_modify_json, method, multiple_methods
                    ),
                    dump_bypass_report_json=_method_output_path(
                        args.dump_bypass_report_json,
                        method,
                        multiple_methods,
                    ),
                )
                dump_context(ctx, quant_output)
                selected_summary = ", ".join(
                    f"{item['kind']}:{item['layer']}="
                    f"{item['abs_max']:.6g}"
                    for item in report["selected"]
                )
                print(
                    f"[BYPASS] method={method} top_k="
                    f"{args.outlier_bypass_top_k} fp16={args.fp16_export} "
                    f"selected=[{selected_summary}]"
                )
            generated_quant_paths[method] = quant_output

    if args.compare_tfdl_quant_fp:
        if args.compare_tfdl_quant_fp == "__generated__":
            quant_targets = generated_quant_paths
        else:
            quant_targets = {
                "external": str(args.compare_tfdl_quant_fp)
            }
        quant_results: dict[str, Any] = {}
        for method, quant_path in quant_targets.items():
            quant_results[method] = compare_tfdl_quant_fp(
                config,
                args.model_path,
                quant_path,
                output_json=None,
                calib_dir=args.calib_dir,
                num_samples=args.num_calib,
                device_name=args.device,
                addon_path=args.addon_path,
            )
        comparison_results["tfdl_quant"] = quant_results

    if args.compare_json and comparison_results:
        Path(args.compare_json).write_text(
            json.dumps(comparison_results, indent=2, sort_keys=True)
        )
    print(
        f"[OK] {args.arch} graph flow prepared: image={config.image_size}, "
        f"seq_len={config.seq_len}, seq_map_hw={config.seq_map_hw}, "
        f"layers={config.num_hidden_layers}, methods={range_methods}"
    )


if __name__ == "__main__":
    main()
