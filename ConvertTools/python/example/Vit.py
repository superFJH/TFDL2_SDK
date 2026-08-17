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

Mixed-precision/quantization design
-----------------------------------
``explicit_qdq`` means that precision boundaries are part of the source graph,
not inserted later with ``TFContext.Modify``.  The intended fast path is:

* UINT8 patch/QKV/attention/MLP projections;
* FP16 prefix/register tokens, LayerNorm outputs and residual additions;
* FP32 LayerNorm gamma/beta (DINOv3 register-token outliers are sensitive to
  rounding these learned parameters to FP16);
* a small range-ranked set of Attention/MLP branches in FP16;
* ``QuantizeLite`` for weight/range conversion, with scalar attention scaling
  represented explicitly by a UINT8 ``Requantize`` lookup table.

The main future optimization knobs are therefore the number of floating Top-K
branches, attention range granularity (whole tensor versus per head), and how
much of the FP16 residual/LayerNorm path the NPU can execute without extra
layout conversions.  Keep every change measurable with CLS and patch-token
cosine: a high aggregate tensor cosine can hide register-token failures.
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


def apply_fp16_residual_reparameterization(
    weights: dict[str, np.ndarray],
    config: ViTOpConfig,
    scale: float,
) -> None:
    """Start a scaled residual stream at block 0's MLP merge.

    DINOv3 register-token residuals can exceed the finite FP16 range after the
    first MLP. Scaling the block-0 residual merge and every later branch update
    by the same positive factor preserves the pre-norm transformer function
    (up to the LayerNorm epsilon) while keeping the residual stream finite.

    The block-0 attention path remains unscaled. Its residual is scaled at the
    block-0 MLP merge, so only block-0 FC2 and later output projections need
    their weights and biases folded by ``scale``.
    """
    scale = float(scale)
    if not (0.0 < scale <= 1.0):
        raise ValueError("FP16 residual scale must be in (0, 1]")
    if scale == 1.0:
        return
    weights["fp16_residual_scale.weight"] = np.asarray(
        [scale], dtype=np.float32
    )
    weights["fp16_residual_scale.bias"] = np.zeros((1,), dtype=np.float32)
    for layer_id in range(config.num_hidden_layers):
        stems = ("fc2",) if layer_id == 0 else ("proj", "fc2")
        for stem in stems:
            for suffix in ("weight", "bias"):
                name = f"layers.{layer_id}.{stem}.{suffix}"
                weights[name] = _as_float32(weights[name] * scale)


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
        # Attention Softmax is logically a collection of H*S independent
        # rows. Preserve one min/max pair per row instead of later expanding a
        # coarse per-head range S times. These arrays stay small compared with
        # the H*S*S attention tensor and are aggregated elementwise over all
        # calibration images.
        self.row_ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
        if name.endswith((".qk_matmul", ".attn_scores")):
            if detached.ndim == 4:
                # [B,H,S,S] -> [H,S], aggregating the calibration batch and
                # key dimension while retaining the runtime H*S row order.
                row_min_tensor = detached.amin(dim=(0, 3))
                row_max_tensor = detached.amax(dim=(0, 3))
            elif detached.ndim == 3:
                # [H,S,S] -> [H,S]
                row_min_tensor = detached.amin(dim=2)
                row_max_tensor = detached.amax(dim=2)
            else:
                raise ValueError(
                    f"{name} row calibration expects rank 3/4, got "
                    f"shape={tuple(detached.shape)}"
                )
            row_min = row_min_tensor.float().cpu().numpy().reshape(-1)
            row_max = row_max_tensor.float().cpu().numpy().reshape(-1)
            row_min = np.minimum(row_min, 0.0).astype(np.float32, copy=False)
            row_max = np.maximum(row_max, 0.0).astype(np.float32, copy=False)
            if name in self.row_ranges:
                old_min, old_max = self.row_ranges[name]
                if old_min.shape != row_min.shape:
                    raise ValueError(
                        f"{name} row count changed during calibration: "
                        f"{old_min.size} -> {row_min.size}"
                    )
                row_min = np.minimum(old_min, row_min)
                row_max = np.maximum(old_max, row_max)
            equal = row_min == row_max
            if np.any(equal):
                epsilon = np.maximum(np.abs(row_min[equal]), 1.0) * 1e-6
                row_min[equal] -= epsilon
                row_max[equal] += epsilon
            self.row_ranges[name] = (row_min, row_max)
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
        output = {
            name: {
                "min": qmin,
                "max": qmax,
                "range_method": self.method,
                "exclude_register_tokens": self.exclude_register_tokens,
                "register_range_policy": self.register_range_policy,
            }
            for name, (qmin, qmax) in sorted(self.resolved_ranges().items())
        }
        for name, (row_min, row_max) in sorted(self.row_ranges.items()):
            output[f"{name}.rows"] = {
                "min": row_min.tolist(),
                "max": row_max.tolist(),
                "range_method": "per-row-minmax",
                "channel_layout": "H*S",
                "row_count": int(row_min.size),
                "calibration_observations": "elementwise union",
                "exclude_register_tokens": False,
                "register_range_policy": self.register_range_policy,
            }
        return output


class TorchViTOpGraph(torch.nn.Module):
    def __init__(self, config: ViTOpConfig, weights: dict[str, np.ndarray], device: torch.device):
        super().__init__()
        self.config = config
        self.weights = {k: torch.from_numpy(v).to(device) for k, v in weights.items()}
        self.collector = RangeCollector()
        self.register_residual_scale = 1.0
        self.fp16_residual_scale = 1.0
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
        resid2_base = hidden
        if layer_id == 0 and float(self.fp16_residual_scale) != 1.0:
            resid2_base = self._record(
                f"{c}.resid2_base_scaled",
                hidden * float(self.fp16_residual_scale),
            )
        return self._record(f"{c}.resid2", resid2_base + mlp)

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
    fp16_residual_scale: float = 1.0,
    stable_attention_window: float = 0.0,
) -> dict[str, dict[str, Any]]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TorchViTOpGraph(config, weights, device).eval()
    model.register_residual_scale = float(register_residual_scale)
    model.fp16_residual_scale = float(fp16_residual_scale)
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
    fp16_residual_scale: float = 1.0,
    conv_native_layout: bool = False,
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
        if torch_op is not None:
            torch_op.fp16_residual_scale = float(fp16_residual_scale)
        samples = _load_compare_samples(config, calib_dir, max(1, num_samples), device)

        tfdl_executor = None
        if compare_tfdl_fp:
            if addon_path is not None:
                _load_addon_if_needed(config, addon_path)
            _, tfdl_executor, _, _, _ = build_vit_tfdl_graph(
                config,
                weights,
                create_executor=True,
                fp16_residual_scale=fp16_residual_scale,
                conv_native_layout=conv_native_layout,
            )

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
        "input_contract": None,
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
        input_is_uint8 = "UINT8" in str(inputs[0].dtype)
        result["input_contract"] = (
            "resized_raw_uint8_nchw"
            if input_is_uint8
            else "resized_raw_float32_nchw"
        )
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

                runtime_input = (
                    raw_uint8
                    if input_is_uint8
                    else np.ascontiguousarray(raw_uint8, dtype=np.float32)
                )
                inputs[0].fromNumpy(runtime_input)
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
            qmin, qmax = value["min"], value["max"]
            # H*S row arrays are loaded separately. Keeping this function
            # scalar-only protects all existing ranking/range consumers.
            if isinstance(qmin, list) or isinstance(qmax, list):
                continue
            out[name] = (float(qmin), float(qmax))
        else:
            out[name] = (float(value[0]), float(value[1]))
    return out


def _load_row_range_json(
    path: str | Path,
) -> dict[str, tuple[list[float], list[float]]]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, tuple[list[float], list[float]]] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            continue
        qmin, qmax = value.get("min"), value.get("max")
        if not isinstance(qmin, list) and not isinstance(qmax, list):
            continue
        if not isinstance(qmin, list) or not isinstance(qmax, list):
            raise ValueError(f"{name}: row min/max must both be arrays")
        if not qmin or len(qmin) != len(qmax):
            raise ValueError(
                f"{name}: row min/max lengths must be equal and non-zero"
            )
        row_min = [float(value) for value in qmin]
        row_max = [float(value) for value in qmax]
        if not np.all(np.isfinite(row_min)) or not np.all(
            np.isfinite(row_max)
        ):
            raise ValueError(f"{name}: non-finite per-row range")
        if any(lo >= hi for lo, hi in zip(row_min, row_max)):
            raise ValueError(f"{name}: every row range must satisfy min < max")
        out[name] = (row_min, row_max)
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
        symbol_tag = name[:-5] if name.endswith(".rows") else name
        if symbol_tag in symbol_map:
            item["tfdl_name"] = symbol_map[symbol_tag]
        annotated[name] = item
    path.write_text(json.dumps(annotated, indent=2, sort_keys=True))


def _tokens_to_conv1d(
    hidden,
    config: ViTOpConfig,
    *,
    conv_native_layout: bool = False,
):
    from TFDL2 import Op

    seq_h, seq_w = config.seq_map_hw
    if conv_native_layout:
        return Op.Reshape(
            hidden,
            (1, config.hidden_size, seq_h, seq_w),
        )
    return Op.Reshape(Op.Transpose(hidden, (0, 2, 1)), (1, config.hidden_size, seq_h, seq_w))


def _conv1d_to_tokens(
    hidden_4d,
    channels: int,
    config: ViTOpConfig,
    *,
    conv_native_layout: bool = False,
):
    from TFDL2 import Op

    if conv_native_layout:
        return Op.Reshape(hidden_4d, (1, channels, config.seq_len))
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


def _split_token_groups(
    hidden,
    config: ViTOpConfig,
    *,
    conv_native_layout: bool = False,
):
    from TFDL2 import Op

    specs = _token_group_specs(config)
    token_axis = 2 if conv_native_layout else 1
    return specs, tuple(
        Op.Slice(
            hidden,
            axis=token_axis,
            split=tuple(size for _, size in specs),
        )
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
    *,
    conv_native_layout: bool = False,
) -> dict[str, Any]:
    from TFDL2 import Op

    specs, groups = _split_token_groups(
        hidden,
        config,
        conv_native_layout=conv_native_layout,
    )
    outputs: dict[str, Any] = {}
    for (group_name, token_count), group in zip(specs, groups):
        channels = int(
            ctx.GetParamSymbol(weight_name).shape[1]
            if hasattr(ctx.GetParamSymbol(weight_name), "shape")
            else config.hidden_size
        )
        group_4d = Op.Reshape(
            group
            if conv_native_layout
            else Op.Transpose(group, (0, 2, 1)),
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
        out_tokens = Op.Reshape(
            out_4d,
            (1, out_channels, token_count),
        )
        outputs[group_name] = (
            out_tokens
            if conv_native_layout
            else Op.Transpose(out_tokens, (0, 2, 1))
        )
    return outputs


def _scale_register_update_op(
    hidden,
    config: ViTOpConfig,
    scale: float,
    symbol_map: dict[str, str],
    tag: str,
    *,
    conv_native_layout: bool = False,
):
    from TFDL2 import Op

    if float(scale) == 1.0 or config.num_register_tokens == 0:
        return hidden
    specs, groups = _split_token_groups(
        hidden,
        config,
        conv_native_layout=conv_native_layout,
    )
    scaled = []
    for (group_name, _), group in zip(specs, groups):
        if group_name == "registers":
            group = Op.Mul(group, float(scale))
        scaled.append(group)
    return _mark(
        symbol_map,
        tag,
        Op.Concat(tuple(scaled), axis=2 if conv_native_layout else 1),
    )


def _qkv_conv_to_heads(
    hidden_4d,
    config: ViTOpConfig,
    *,
    conv_native_layout: bool = False,
):
    from TFDL2 import Op

    heads = Op.Reshape(hidden_4d, (1, config.num_attention_heads, config.head_dim, -1))
    if conv_native_layout:
        return heads
    return Op.Transpose(heads, (0, 1, 3, 2))


def _attention_to_conv1d(attn_3d, config: ViTOpConfig):
    from TFDL2 import Op

    seq_h, seq_w = config.seq_map_hw
    attn_4d = Op.Reshape(attn_3d, (1, config.num_attention_heads, config.seq_len, config.head_dim))
    attn_4d = Op.Transpose(attn_4d, (0, 1, 3, 2))
    return Op.Reshape(attn_4d, (1, config.hidden_size, seq_h, seq_w))


def _apply_rope_op(
    q,
    k,
    rope_sin,
    rope_cos,
    layer_id: int,
    *,
    conv_native_layout: bool = False,
):
    from TFDL2 import Op

    custom_param = (
        json.dumps(
            {
                "inputLayout": "BHDN",
                "qOutputLayout": "BHND",
                "kOutputLayout": "BHDN",
            },
            separators=(",", ":"),
        )
        if conv_native_layout
        else "{}"
    )
    out = Op.Custom(
        (q, k, rope_sin, rope_cos),
        (f"vit_q_rope_{layer_id}", f"vit_k_rope_{layer_id}"),
        "ApplyRope",
        custom_param,
    )
    return out[0], out[1]


def _linear_requant_maptable(
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    *,
    scale: float,
    bias: float = 0.0,
) -> list[int]:
    """Encode ``y = x * scale + bias`` as a UINT8 -> UINT8 lookup table.

    TFDL activation quantization is affine and per tensor here.  Reconstruct
    each possible input code with the input scale/zero-point, apply the scalar
    operation in this offline builder, then encode with the output
    scale/zero-point.  The runtime operator is consequently only a 256-byte
    lookup.  This is the source-graph equivalent of the full Quant pass folding
    a tensor-wise Mul/Div into Requant.

    For QK scaling the calibrated output range is normally the input range
    divided by ``sqrt(head_dim)``.  The table can then be numerically identity
    while still changing qscale; do not remove it as a redundant identity op.
    """
    input_min, input_max = (float(value) for value in input_range)
    output_min, output_max = (float(value) for value in output_range)
    if not (
        np.isfinite(input_min)
        and np.isfinite(input_max)
        and np.isfinite(output_min)
        and np.isfinite(output_max)
        and np.isfinite(scale)
        and np.isfinite(bias)
        and input_min < input_max
        and output_min < output_max
    ):
        raise ValueError(
            "invalid ranges/scalars for Requantize map table: "
            f"input={input_range}, output={output_range}, "
            f"scale={scale}, bias={bias}"
        )
    input_step = (input_max - input_min) / 255.0
    output_step = (output_max - output_min) / 255.0
    input_zero = float(np.clip(np.rint(-input_min / input_step), 0.0, 255.0))
    output_zero = float(
        np.clip(np.rint(-output_min / output_step), 0.0, 255.0)
    )
    input_codes = np.arange(256, dtype=np.float64)
    real_input = (input_codes - input_zero) * input_step
    real_output = real_input * float(scale) + float(bias)
    output_codes = np.rint(real_output / output_step + output_zero)
    return np.clip(output_codes, 0.0, 255.0).astype(np.uint8).tolist()


def _build_block_op(
    ctx,
    hidden,
    config: ViTOpConfig,
    layer_id: int,
    symbol_map: dict[str, str],
    rope_sin=None,
    rope_cos=None,
    register_residual_scale: float = 1.0,
    fp16_residual_scale: float = 1.0,
    split_mlp_token_groups: bool = False,
    split_attention_heads: bool = False,
    stable_attention_window: float = 0.0,
    explicit_qdq: bool = False,
    fp_attn_layers: frozenset[int] = frozenset(),
    fp_mlp_layers: frozenset[int] = frozenset(),
    dequant_dtype: Any = None,
    float_entry_tensors: list[str] | None = None,
    attention_requant_maptable: list[int] | dict[int, list[int]] | None = None,
    source_quantize_entries: bool = False,
    per_channel_qk: bool = False,
    fold_attention_scale_into_q: bool = False,
    conv_native_layout: bool = False,
):
    """Build one transformer block with source-level precision boundaries.

    ``fp_attn_layers`` keeps the output projection floating; the QK/Softmax/AV
    core stays quantized. ``fp_mlp_layers`` keeps the complete MLP floating.
    ``per_channel_qk`` keeps QK quantized with one range per Softmax row.  Its
    Requantize output receives the same H*S qinfo divided by
    ``sqrt(head_dim)``; the lookup table is the exact ``0..255`` identity map.
    ``fold_attention_scale_into_q`` instead scales the Q projection offline
    and connects QK directly to UINT8 Softmax. Softmax is converted back to
    UINT8 with ``Context.Modify`` after all qinfo is registered.
    Every other branch is dequantized only at its residual Add.  This division
    preserves the high-throughput matrix work on NPU while keeping the
    numerically sensitive residual stream in FP16/FP32.
    """
    from TFDL2 import Op
    from TFDL2.Common import TFDataType

    if dequant_dtype is None:
        dequant_dtype = TFDataType.TFDL_FLOAT
    c = f"layers.{layer_id}"

    def dequantize(value: Any, tag: str) -> Any:
        # Record the *floating* side as a stop tensor.  QuantizeLite leaves the
        # node untouched; full Quant must also be prevented from propagating
        # UINT8 through it and silently changing the residual-path contract.
        value = Op.DeQuantize(value, dequant_dtype)
        if float_entry_tensors is not None:
            float_entry_tensors.append(str(value))
        return _mark(symbol_map, tag, value)

    attention_scale: Any = 1.0 / math.sqrt(config.head_dim)
    if (
        explicit_qdq
        and dequant_dtype == TFDataType.TFDL_FLOAT16
        and attention_requant_maptable is None
        and not per_channel_qk
        and not fold_attention_scale_into_q
    ):
        attention_scale = ctx.GetParamSymbol(f"{c}.attn_scale")

    norm_axis = 1 if conv_native_layout else -1
    normed = Op.LayerNorm2(hidden, ctx.GetParamSymbol(f"{c}.norm1.weight"), ctx.GetParamSymbol(f"{c}.norm1.bias"), axis=norm_axis)
    normed = _mark(symbol_map, f"{c}.norm1", normed)
    normed_for_qkv = normed
    if source_quantize_entries:
        # QuantizeLite performs no propagation or graph rewrite.  This source
        # Quantize is therefore the explicit entry to Q/K/V's UINT8 island.
        normed_for_qkv = _mark(
            symbol_map,
            f"{c}.norm1.quantized",
            Op.Quantize(normed_for_qkv),
        )
    norm1_transposed = (
        normed_for_qkv
        if conv_native_layout
        else _mark(
            symbol_map,
            f"{c}.norm1_transposed",
            Op.Transpose(normed_for_qkv, (0, 2, 1)),
        )
    )
    normed_4d = _mark(
        symbol_map,
        f"{c}.norm1_conv_input",
        Op.Reshape(
            norm1_transposed,
            (1, config.hidden_size, *config.seq_map_hw),
        ),
    )
    q = _qkv_conv_to_heads(
        _pointwise_4d_op(ctx, normed_4d, f"{c}.q.weight", f"{c}.q.bias", config.hidden_size, symbol_map, f"{c}.q"),
        config,
        conv_native_layout=conv_native_layout,
    )
    k = _qkv_conv_to_heads(
        _pointwise_4d_op(ctx, normed_4d, f"{c}.k.weight", f"{c}.k.bias", config.hidden_size, symbol_map, f"{c}.k"),
        config,
        conv_native_layout=conv_native_layout,
    )
    v = _qkv_conv_to_heads(
        _pointwise_4d_op(ctx, normed_4d, f"{c}.v.weight", f"{c}.v.bias", config.hidden_size, symbol_map, f"{c}.v"),
        config,
        conv_native_layout=conv_native_layout,
    )
    if config.use_rope:
        q, k = _apply_rope_op(
            q,
            k,
            rope_sin,
            rope_cos,
            layer_id,
            conv_native_layout=conv_native_layout,
        )
        _mark(symbol_map, f"{c}.q_rope", q)
        _mark(symbol_map, f"{c}.k_rope", k)
    elif conv_native_layout:
        # Non-RoPE ViTs still need Q in row-major [H,N,D]. K can remain in
        # the Conv-native [H,D,N] layout required by QK MatMul.
        q = Op.Transpose(q, (0, 1, 3, 2))
    if conv_native_layout:
        # AV remains exactly prob @ V. Materialize V as [H,N,D] because NPU
        # MatMul trans flags compile back into explicit Transpose operators.
        v = Op.Transpose(v, (0, 1, 3, 2))
    q3 = Op.Reshape(q, (config.num_attention_heads, config.seq_len, config.head_dim))
    k3 = (
        Op.Reshape(k, (config.num_attention_heads, config.head_dim, config.seq_len))
        if conv_native_layout
        else Op.Transpose(
            Op.Reshape(
                k,
                (
                    config.num_attention_heads,
                    config.seq_len,
                    config.head_dim,
                ),
            ),
            (0, 2, 1),
        )
    )
    v3 = Op.Reshape(v, (config.num_attention_heads, config.seq_len, config.head_dim))
    _mark(symbol_map, f"{c}.q_matmul_input", q3)
    _mark(symbol_map, f"{c}.k_matmul_input", k3)
    _mark(symbol_map, f"{c}.v_matmul_input", v3)
    if split_attention_heads:
        # Experimental accuracy knob: separate heads can use independent QK
        # and score ranges.  It increases Slice/Concat overhead substantially,
        # so validate end-to-end latency and Softmax accuracy before enabling.
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
            head_requant_maptable = (
                attention_requant_maptable.get(head_id)
                if isinstance(attention_requant_maptable, dict)
                else attention_requant_maptable
            )
            scaled_head = (
                # Scaling Q before RoPE is equivalent to scaling QK after
                # MatMul, so no score-side operator is needed in fold-Q mode.
                qk_head
                if fold_attention_scale_into_q
                # Requant replaces QK * (1/sqrt(head_dim)) without visiting a
                # floating tensor. Its output qinfo belongs to attn_scores.
                else Op.Requantize(qk_head, head_requant_maptable)
                if head_requant_maptable is not None
                else Op.Mul(qk_head, attention_scale)
            )
            scores_head = _mark(
                symbol_map, f"{c}.attn_scores.raw.{h}", scaled_head
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
        if fold_attention_scale_into_q:
            # q.weight/q.bias and their activation ranges were scaled by
            # 1/sqrt(head_dim), so this UINT8 MatMul already represents the
            # scaled attention scores. This applies to both scalar and H*S
            # QK qinfo; no runtime Requant/DQ/Div is needed.
            scaled_scores = qk_matmul
        elif per_channel_qk:
            # The score qinfo is registered as the QK qinfo divided by
            # sqrt(head_dim).  Therefore each UINT8 code already represents
            # the correctly scaled score and Requantize is deliberately the
            # exact identity map.  Unlike fold-Q, this leaves Q/K parameters
            # and their activation codes unchanged.  QK codes intentionally
            # use the new H*S qinfo (so they are not scalar-baseline-identical)
            # while Softmax receives one correctly scaled qinfo per score row.
            scaled_scores = Op.Requantize(qk_matmul, list(range(256)))
        else:
            scaled_scores = (
                # QuantizeLite intentionally will not discover/fold scalar
                # Mul/Div. Express that fold directly so QK and Softmax stay
                # UINT8 in the ordinary per-tensor path.
                Op.Requantize(qk_matmul, attention_requant_maptable)
                if attention_requant_maptable is not None
                else Op.Mul(qk_matmul, attention_scale)
            )
        scores = _mark(symbol_map, f"{c}.attn_scores", scaled_scores)
        probs = _mark(symbol_map, f"{c}.attn_probs", Op.Softmax(scores, axis=2))
        attn = _mark(
            symbol_map,
            f"{c}.av_matmul",
            Op.MatMul(probs, v3, transA=False, transB=False),
        )
    attn_4d = _mark(symbol_map, f"{c}.attn", _attention_to_conv1d(attn, config))
    if explicit_qdq and layer_id in fp_attn_layers:
        # Top-K attention bypass starts after AV: QKV/QK/Softmax/AV remain
        # INT8, while only the selected output projection uses FP weights.
        attn_4d = dequantize(attn_4d, f"{c}.attn.dequantized")
    proj = _mark(
        symbol_map,
        f"{c}.proj_tokens",
        _conv1d_to_tokens(
        _pointwise_4d_op(ctx, attn_4d, f"{c}.proj.weight", f"{c}.proj.bias", config.hidden_size, symbol_map, f"{c}.proj"),
        config.hidden_size,
        config,
        conv_native_layout=conv_native_layout,
        ),
    )
    proj = _scale_register_update_op(
        proj,
        config,
        register_residual_scale,
        symbol_map,
        f"{c}.proj.scaled",
        conv_native_layout=conv_native_layout,
    )
    if explicit_qdq and layer_id not in fp_attn_layers:
        # The normal INT8 projection exits immediately before residual Add.
        proj = dequantize(proj, f"{c}.proj_tokens.dequantized")
    hidden = _mark(symbol_map, f"{c}.resid1", Op.Add(hidden, proj))
    normed2 = _mark(symbol_map, f"{c}.norm2", Op.LayerNorm2(hidden, ctx.GetParamSymbol(f"{c}.norm2.weight"), ctx.GetParamSymbol(f"{c}.norm2.bias"), axis=norm_axis))
    normed2_for_mlp = normed2
    if source_quantize_entries and layer_id not in fp_mlp_layers:
        # Explicit entry to the MLP INT8 island.  A selected Top-K MLP skips
        # this Quantize and keeps FC1/activation/FC2 together in FP16.
        normed2_for_mlp = _mark(
            symbol_map,
            f"{c}.norm2.quantized",
            Op.Quantize(normed2_for_mlp),
        )
    norm2_transposed = (
        normed2_for_mlp
        if conv_native_layout
        else _mark(
            symbol_map,
            f"{c}.norm2_transposed",
            Op.Transpose(normed2_for_mlp, (0, 2, 1)),
        )
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
                conv_native_layout=conv_native_layout,
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
                    mid
                    if conv_native_layout
                    else Op.Transpose(mid, (0, 2, 1)),
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
                out = Op.Reshape(
                    raw,
                    (1, config.hidden_size, token_count),
                )
                if not conv_native_layout:
                    out = Op.Transpose(out, (0, 2, 1))
                if group_name == "registers" and float(register_residual_scale) != 1.0:
                    out = Op.Mul(out, float(register_residual_scale))
                out = _mark(symbol_map, f"{c}.fc2.{group_name}", out)
                fc2_groups.append(out)
            mlp = _mark(
                symbol_map,
                f"{c}.fc2.scaled",
                Op.Concat(
                    tuple(fc2_groups),
                    axis=2 if conv_native_layout else 1,
                ),
            )
            symbol_map[f"{c}.fc2"] = str(mlp)
            resid2_base = hidden
            if layer_id == 0 and float(fp16_residual_scale) != 1.0:
                resid2_base = _mark(
                    symbol_map,
                    f"{c}.resid2_base_scaled",
                    Op.Scale2(
                        hidden,
                        ctx.GetParamSymbol("fp16_residual_scale.weight"),
                        ctx.GetParamSymbol("fp16_residual_scale.bias"),
                    ),
                )
            return _mark(
                symbol_map,
                f"{c}.resid2",
                Op.Add(resid2_base, mlp),
            )
        mlp = _pointwise_4d_op(ctx, normed2_4d, f"{c}.fc1.weight", f"{c}.fc1.bias", config.intermediate_size, symbol_map, f"{c}.fc1")
        mlp = _mark(symbol_map, f"{c}.mlp_mid", Op.GeLU(mlp))
    mlp = _mark(
        symbol_map,
        f"{c}.fc2_tokens",
        _conv1d_to_tokens(
        _pointwise_4d_op(ctx, mlp, f"{c}.fc2.weight", f"{c}.fc2.bias", config.hidden_size, symbol_map, f"{c}.fc2"),
        config.hidden_size,
        config,
        conv_native_layout=conv_native_layout,
        ),
    )
    if explicit_qdq and layer_id not in fp_mlp_layers:
        # Keep FC2 in UINT8 and convert only its result for the residual Add.
        mlp = dequantize(mlp, f"{c}.fc2_tokens.dequantized")
    resid2_base = hidden
    if layer_id == 0 and float(fp16_residual_scale) != 1.0:
        # DINOv3's first MLP can create a ~4e4 register-token outlier.  This
        # exact reparameterization protects FP16 headroom; paired weights and
        # ranges are prepared by the loader/collector, so do not tune it only
        # on this Add without regenerating calibration data.
        resid2_base = _mark(
            symbol_map,
            f"{c}.resid2_base_scaled",
            Op.Scale2(
                hidden,
                ctx.GetParamSymbol("fp16_residual_scale.weight"),
                ctx.GetParamSymbol("fp16_residual_scale.bias"),
            ),
        )
    return _mark(symbol_map, f"{c}.resid2", Op.Add(resid2_base, mlp))


def build_vit_tfdl_graph(
    config: ViTOpConfig,
    weights: dict[str, np.ndarray],
    range_json: str | Path | None = None,
    create_executor: bool = False,
    register_residual_scale: float = 1.0,
    fp16_residual_scale: float = 1.0,
    split_mlp_token_groups: bool = False,
    split_attention_heads: bool = False,
    stable_attention_window: float = 0.0,
    explicit_qdq: bool = False,
    fp_attn_layers: Sequence[int] = (),
    fp_mlp_layers: Sequence[int] = (),
    fp16_export: bool = False,
    attention_scale_requant: bool = False,
    source_quantize_entries: bool = False,
    per_channel_qk: bool = False,
    fold_attention_scale_into_q: bool = False,
    per_row_attention_range_floor: float = 0.0,
    per_channel_qk_max_requant_multiplier: float | None = 0.99,
    conv_native_layout: bool = False,
):
    """Build the float or explicit mixed-precision TFDL source graph.

    This function owns the precision contract.  The calibration pass should
    only encode weights/ranges; it must not be required to discover where the
    graph crosses INT8 and floating domains.  ``symbol_map`` deliberately uses
    semantic names so Torch-collected ranges survive added Q/DQ/reshape nodes.
    """
    from TFDL2 import TFContext, TFExecutor, Op
    from TFDL2.Common import TFDataType

    fp_attn_layer_set = frozenset(int(value) for value in fp_attn_layers)
    fp_mlp_layer_set = frozenset(int(value) for value in fp_mlp_layers)
    dequant_dtype = (
        TFDataType.TFDL_FLOAT16 if fp16_export else TFDataType.TFDL_FLOAT
    )
    loaded_ranges = _load_range_json(range_json) if range_json else None
    loaded_row_ranges = (
        _load_row_range_json(range_json) if range_json else {}
    )
    if attention_scale_requant and loaded_ranges is None:
        raise ValueError("attention Requantize requires --range-json")
    if per_channel_qk and loaded_ranges is None:
        raise ValueError("per-channel QK quantization requires --range-json")
    if per_channel_qk and attention_scale_requant:
        raise ValueError(
            "per-channel QK quantization replaces attention Requantize"
        )
    if per_channel_qk and split_attention_heads:
        raise ValueError(
            "per-channel QK quantization keeps heads in one MatMul and is "
            "incompatible with split_attention_heads"
        )
    if not 0.0 <= float(per_row_attention_range_floor) <= 1.0:
        raise ValueError("per-row attention range floor must be in [0, 1]")
    qk_range_floor = float(per_row_attention_range_floor)
    qk_max_requant_multiplier = (
        None
        if per_channel_qk_max_requant_multiplier is None
        else float(per_channel_qk_max_requant_multiplier)
    )
    if qk_max_requant_multiplier is not None and (
        not np.isfinite(qk_max_requant_multiplier)
        or qk_max_requant_multiplier <= 0.0
    ):
        raise ValueError(
            "per-channel QK max requant multiplier must be positive"
        )
    # Requant mode historically implied source Quantize entries. Keep that
    # behavior for direct callers while allowing the per-channel path to use
    # the same explicit INT8 islands without constructing Requant tables.
    source_quantize_entries = bool(
        source_quantize_entries or attention_scale_requant
    )

    attention_requant_tables: dict[
        int, list[int] | dict[int, list[int]]
    ] = {}
    if attention_scale_requant:
        # Prefer independently observed QK and scaled-score ranges.  Falling
        # back to score_range / scale supports older calibration JSON files
        # which did not expose the pre-scale QK tensor.
        attention_scale_value = 1.0 / math.sqrt(config.head_dim)
        for layer_id in range(config.num_hidden_layers):
            if split_attention_heads:
                head_tables: dict[int, list[int]] = {}
                for head_id in range(config.num_attention_heads):
                    score_tag = (
                        f"layers.{layer_id}.attn_scores.h{head_id:02d}"
                    )
                    if score_tag not in loaded_ranges:
                        raise KeyError(
                            f"range JSON is missing {score_tag!r}, required "
                            "for per-head attention Requantize"
                        )
                    output_range = loaded_ranges[score_tag]
                    qk_tag = f"layers.{layer_id}.qk_matmul.h{head_id:02d}"
                    input_range = loaded_ranges.get(qk_tag)
                    if input_range is None:
                        input_range = (
                            float(output_range[0]) / attention_scale_value,
                            float(output_range[1]) / attention_scale_value,
                        )
                    head_tables[head_id] = _linear_requant_maptable(
                        input_range,
                        output_range,
                        scale=attention_scale_value,
                    )
                attention_requant_tables[layer_id] = head_tables
            else:
                score_tag = f"layers.{layer_id}.attn_scores"
                if score_tag not in loaded_ranges:
                    raise KeyError(
                        f"range JSON is missing {score_tag!r}, required to "
                        "build the attention-scale Requantize map table"
                    )
                output_range = loaded_ranges[score_tag]
                qk_tag = f"layers.{layer_id}.qk_matmul"
                input_range = loaded_ranges.get(qk_tag)
                if input_range is None:
                    input_range = (
                        float(output_range[0]) / attention_scale_value,
                        float(output_range[1]) / attention_scale_value,
                    )
                attention_requant_tables[layer_id] = _linear_requant_maptable(
                    input_range,
                    output_range,
                    scale=attention_scale_value,
                )
    graph_weights = dict(weights)
    if fold_attention_scale_into_q:
        # RoPE is linear, therefore scaling Q before RoPE is exactly
        # equivalent to scaling QK afterward. Scaling real parameters (as
        # opposed to only changing qinfo) preserves the intended semantic
        # factor through quantized Conv and MatMul.
        attention_scale_value = 1.0 / math.sqrt(config.head_dim)
        for layer_id in range(config.num_hidden_layers):
            for suffix in ("q.weight", "q.bias"):
                name = f"layers.{layer_id}.{suffix}"
                value = graph_weights[name]
                graph_weights[name] = np.ascontiguousarray(
                    value * attention_scale_value,
                    dtype=value.dtype,
                )
    if explicit_qdq and fp16_export:
        # Only tensors consumed by known FP16 compute islands are converted.
        # LayerNorm gamma/beta intentionally stay FP32: DINOv3's learned norm
        # parameters plus register outliers made an all-parameter FP16 export
        # collapse block-0 norm2 and the subsequent residual stream.
        fp16_params = {
            "prefix_tokens",
            "pos_embed",
            "fp16_residual_scale.weight",
            "fp16_residual_scale.bias",
        }
        for layer_id in range(config.num_hidden_layers):
            prefix = f"layers.{layer_id}"
            if layer_id in fp_attn_layer_set:
                fp16_params.update(
                    {f"{prefix}.proj.weight", f"{prefix}.proj.bias"}
                )
            if layer_id in fp_mlp_layer_set:
                if config.use_gated_mlp:
                    names = ("gate", "up", "fc2")
                else:
                    names = ("fc1", "fc2")
                for stem in names:
                    fp16_params.update(
                        {f"{prefix}.{stem}.weight", f"{prefix}.{stem}.bias"}
                    )
        graph_weights = {
            name: (
                np.ascontiguousarray(value, dtype=np.float16)
                if name in fp16_params
                else value
            )
            for name, value in graph_weights.items()
        }
        if (
            not attention_scale_requant
            and not per_channel_qk
            and not fold_attention_scale_into_q
        ):
            for layer_id in range(config.num_hidden_layers):
                graph_weights[f"layers.{layer_id}.attn_scale"] = np.asarray(
                    [1.0 / math.sqrt(config.head_dim)], dtype=np.float16
                )

    if conv_native_layout:
        # Residual/token state is [B,C,N]. Constants are transposed offline so
        # the runtime graph does not pay layout conversions around Concat/Add.
        for name in ("prefix_tokens", "pos_embed"):
            if name in graph_weights:
                graph_weights[name] = np.ascontiguousarray(
                    np.transpose(graph_weights[name], (0, 2, 1))
                )
        # Conv-native ApplyRope consumes [B,H,D,N], so tables use [B,H,D,hw]
        # as well. This is an offline parameter transform, not a graph op.
        for name in ("rope_sin", "rope_cos"):
            if name in graph_weights:
                graph_weights[name] = np.ascontiguousarray(
                    np.transpose(graph_weights[name], (0, 1, 3, 2))
                )

    ctx = TFContext(
        f"{config.arch}_vit_op"
        + ("_conv_native" if conv_native_layout else "")
    )
    ctx.RegisterParamToContext(**graph_weights)
    symbol_map: dict[str, str] = {}
    float_entry_tensors: list[str] = []
    with ctx:
        # Placeholder preprocessing turns raw 0..255 RGB into the normalized
        # model domain.  QuantizeLite preserves this FLOAT Placeholder, so its
        # runtime buffer is raw FP32; full Quant converts it to a UINT8 input.
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
        patch_input = pixel_values
        if source_quantize_entries:
            # QuantizeLite does not infer the first quantized region.  Keep an
            # explicit source entry so patch weights are encoded and executed
            # as UINT8 while preprocessing remains visible and floating.
            patch_input = _mark(
                symbol_map,
                "input.normalized.quantized",
                Op.Quantize(patch_input),
            )
        hidden = Op.Convolution2(
            patch_input,
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
        hidden = Op.Reshape(
            hidden,
            (1, config.hidden_size, config.num_patches),
        )
        if not conv_native_layout:
            hidden = Op.Transpose(hidden, (0, 2, 1))
        if explicit_qdq:
            # Prefix/register tokens and residuals share one floating domain.
            # Exit patch projection here rather than quantizing Concat/Add.
            hidden = _mark(symbol_map, "patch.tokens.quantized", hidden)
            hidden = Op.DeQuantize(hidden, dequant_dtype)
            float_entry_tensors.append(str(hidden))
            hidden = _mark(symbol_map, "patch.tokens", hidden)
        else:
            hidden = _mark(symbol_map, "patch.tokens", hidden)
        if config.prefix_len:
            hidden = _mark(
                symbol_map,
                "prefix_concat",
                Op.Concat(
                    (ctx.GetParamSymbol("prefix_tokens"), hidden),
                    axis=2 if conv_native_layout else 1,
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
            conv_native_layout=conv_native_layout,
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
                fp16_residual_scale=fp16_residual_scale,
                split_mlp_token_groups=split_mlp_token_groups,
                split_attention_heads=split_attention_heads,
                stable_attention_window=stable_attention_window,
                explicit_qdq=explicit_qdq,
                fp_attn_layers=fp_attn_layer_set,
                fp_mlp_layers=fp_mlp_layer_set,
                dequant_dtype=dequant_dtype,
                float_entry_tensors=float_entry_tensors,
                attention_requant_maptable=attention_requant_tables.get(
                    layer_id
                ),
                source_quantize_entries=source_quantize_entries,
                per_channel_qk=per_channel_qk,
                fold_attention_scale_into_q=fold_attention_scale_into_q,
                conv_native_layout=conv_native_layout,
            )
        hidden = _mark(
            symbol_map,
            "final_norm",
            Op.LayerNorm2(
                hidden,
                ctx.GetParamSymbol("norm.weight"),
                ctx.GetParamSymbol("norm.bias"),
                axis=1 if conv_native_layout else -1,
            ),
        )
        cls = Op.Gather2(hidden, 0, 2 if conv_native_layout else 1)
        cls = _mark(symbol_map, "output.cls", Op.Reshape(cls, (1, config.hidden_size)))
        output_tokens = (
            Op.Transpose(hidden, (0, 2, 1))
            if conv_native_layout
            else hidden
        )
        output_names = [str(cls), str(output_tokens)]
    ctx.SetOutputs(output_names)

    if range_json:
        ranges = loaded_ranges
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
        def range_for_graph_tag(
            tag: str,
        ) -> tuple[str, tuple[float, float]] | None:
            # Q/DQ and layout-only nodes do not have independent Torch
            # observations.  Walk back to the nearest semantic tensor whose
            # range is valid.  Add mappings here when introducing a new source
            # boundary; missing qinfo often surfaces much later at runtime.
            range_tag = tag
            for suffix in (".quantized", ".dequantized"):
                if range_tag.endswith(suffix):
                    range_tag = range_tag[: -len(suffix)]
                    break
            # In fold-Q mode QK is already the scaled score tensor. Scalar
            # QK needs the calibrated score range here; otherwise registering
            # qk_matmul first would attach the unscaled range to that tensor.
            if (
                fold_attention_scale_into_q
                and not per_channel_qk
                and range_tag.endswith(".qk_matmul")
            ):
                score_tag = range_tag[: -len(".qk_matmul")] + ".attn_scores"
                if score_tag in ranges:
                    return score_tag, ranges[score_tag]
            candidates = [range_tag]
            if ".qk_matmul.h" in range_tag:
                candidates.append(range_tag.split(".qk_matmul.h", 1)[0] + ".qk_matmul")
            if ".attn_scores.raw.h" in range_tag or ".attn_scores.h" in range_tag:
                candidates.append(range_tag.split(".attn_scores", 1)[0] + ".attn_scores")
            if range_tag == "patch.tokens":
                candidates.append("patch.conv")
            if range_tag.endswith(".proj_tokens"):
                candidates.append(range_tag[: -len("_tokens")])
            if range_tag.endswith(".fc2_tokens"):
                candidates.append(range_tag[: -len("_tokens")])
            if range_tag.endswith(".av_matmul"):
                candidates.append(
                    range_tag[: -len("av_matmul")] + "attn"
                )
            if range_tag.endswith(".q_matmul_input"):
                prefix = range_tag[: -len(".q_matmul_input")]
                candidates.append(
                    f"{prefix}.q_rope" if config.use_rope else f"{prefix}.q"
                )
            for candidate in candidates:
                if candidate in ranges:
                    return candidate, ranges[candidate]
            if range_tag.endswith(".qk_matmul") or ".qk_matmul.h" in range_tag:
                score_tag = (
                    range_tag.split(".qk_matmul", 1)[0] + ".attn_scores"
                )
                if score_tag in ranges:
                    qmin, qmax = ranges[score_tag]
                    scale = 1.0 / math.sqrt(config.head_dim)
                    return score_tag, (qmin / scale, qmax / scale)
            return None

        per_row_attention_configs: dict[
            str, tuple[list[float], list[float]]
        ] = {}
        if per_channel_qk:
            expected_rows = config.num_attention_heads * config.seq_len
            attention_scale_value = 1.0 / math.sqrt(config.head_dim)
            for layer_id in range(config.num_hidden_layers):
                qk_tag = f"layers.{layer_id}.qk_matmul"
                qk_name = symbol_map[qk_tag]
                qk_range_tag = (
                    f"layers.{layer_id}.attn_scores.rows"
                    if fold_attention_scale_into_q
                    else f"layers.{layer_id}.qk_matmul.rows"
                )
                if qk_range_tag not in loaded_row_ranges:
                    raise KeyError(
                        "--per-channel-qk now requires true H*S row ranges; "
                        f"missing {qk_range_tag}. Regenerate --range-json with "
                        "the current Vit.py --dump-minmax-json."
                    )
                qk_min, qk_max = loaded_row_ranges[qk_range_tag]
                if len(qk_min) != expected_rows:
                    raise ValueError(
                        f"{qk_range_tag}: got {len(qk_min)} rows, expected "
                        f"H*S={config.num_attention_heads}*"
                        f"{config.seq_len}={expected_rows}"
                    )
                if qk_range_floor > 0.0:
                    qk_head_stem = (
                        f"layers.{layer_id}.attn_scores"
                        if fold_attention_scale_into_q
                        else f"layers.{layer_id}.qk_matmul"
                    )
                    for head_id in range(config.num_attention_heads):
                        start = head_id * config.seq_len
                        end = start + config.seq_len
                        qk_head_min, qk_head_max = ranges[
                            f"{qk_head_stem}.h{head_id:02d}"
                        ]
                        qk_floor_min = qk_head_min * qk_range_floor
                        qk_floor_max = qk_head_max * qk_range_floor
                        qk_min[start:end] = [
                            min(value, qk_floor_min)
                            for value in qk_min[start:end]
                        ]
                        qk_max[start:end] = [
                            max(value, qk_floor_max)
                            for value in qk_max[start:end]
                        ]
                if qk_max_requant_multiplier is not None:
                    # MatMul's INT32 accumulator represents one integer step
                    # as Q_scale*K_scale. gemmlowp's
                    # QuantizeMultiplierSmallerThanOne requires the output
                    # postscale multiplier to be strictly below one. Expand
                    # ranges whose multiplier exceeds the requested limit,
                    # preserving each range's min/max ratio and zero point.
                    # The default 0.99 leaves an explicit margin for float32
                    # range/qscale round trips during FB serialization.
                    q_input_tag = (
                        f"layers.{layer_id}.q_rope"
                        if config.use_rope
                        else f"layers.{layer_id}.q"
                    )
                    k_input_tag = (
                        f"layers.{layer_id}.k_rope"
                        if config.use_rope
                        else f"layers.{layer_id}.k"
                    )
                    q_input_min, q_input_max = ranges[q_input_tag]
                    k_input_min, k_input_max = ranges[k_input_tag]
                    if fold_attention_scale_into_q:
                        q_input_min *= attention_scale_value
                        q_input_max *= attention_scale_value
                    accumulator_scale = (
                        (q_input_max - q_input_min)
                        * (k_input_max - k_input_min)
                        / (255.0 * 255.0)
                    )
                    minimum_row_scale = float(
                        np.nextafter(
                            np.float32(
                                accumulator_scale
                                / qk_max_requant_multiplier
                            ),
                            np.float32(np.inf),
                        )
                    )
                    qk_min_array = np.asarray(qk_min, dtype=np.float64)
                    qk_max_array = np.asarray(qk_max, dtype=np.float64)
                    row_scales = (qk_max_array - qk_min_array) / 255.0
                    expand_mask = row_scales < minimum_row_scale
                    expanded_rows = int(np.count_nonzero(expand_mask))
                    maximum_multiplier_before = float(
                        np.max(accumulator_scale / row_scales)
                    )
                    if expanded_rows:
                        factors = np.ones_like(row_scales)
                        factors[expand_mask] = (
                            minimum_row_scale / row_scales[expand_mask]
                        )
                        qk_min_array *= factors
                        qk_max_array *= factors
                        qk_min = qk_min_array.astype(np.float32).tolist()
                        qk_max = qk_max_array.astype(np.float32).tolist()
                    maximum_multiplier_after = float(
                        np.max(
                            accumulator_scale
                            / ((qk_max_array - qk_min_array) / 255.0)
                        )
                    )
                    print(
                        f"[QK-SCALE-FLOOR] layer={layer_id:02d} "
                        f"expanded={expanded_rows}/{expected_rows} "
                        f"max_multiplier={maximum_multiplier_before:.6g}->"
                        f"{maximum_multiplier_after:.6g}"
                    )
                per_row_attention_configs[qk_name] = (qk_max, qk_min)
                # Do not calibrate this from an independent observation: this
                # is precisely the qinfo transform performed by the scalar
                # baseline Requant. Keeping it derived from QK makes the
                # 256-entry LUT an exact identity map for every score row
                # while its decoded value is /sqrt(head_dim).
                score_name = symbol_map[f"layers.{layer_id}.attn_scores"]
                per_row_attention_configs[score_name] = (
                    [value * attention_scale_value for value in qk_max],
                    [value * attention_scale_value for value in qk_min],
                )

        for tag, actual in symbol_map.items():
            # The same QK symbol is also tagged as attn_scores in the folded
            # graph. Do not overwrite its vector qinfo with a scalar range.
            if actual in per_row_attention_configs:
                continue
            resolved_range = range_for_graph_tag(tag)
            if resolved_range is not None:
                range_tag, (qmin, qmax) = resolved_range
                if fold_attention_scale_into_q and (
                    tag.endswith(".q")
                    or tag.endswith(".q_rope")
                    or tag.endswith(".q_matmul_input")
                ):
                    qmin *= attention_scale_value
                    qmax *= attention_scale_value
                if not ctx.AddInt8Config(actual, float(qmax), float(qmin)):
                    raise RuntimeError(
                        f"failed to add int8 config for {tag} "
                        f"(range {range_tag}) -> {actual}"
                    )
        for actual, (row_max, row_min) in per_row_attention_configs.items():
            if not ctx.AddInt8ConfigPerChannel(actual, row_max, row_min):
                raise RuntimeError(
                    "failed to add per-row attention int8 config for "
                    f"{actual} ({len(row_max)} rows)"
                )
        if per_channel_qk:
            # Softmax still needs one ordinary output qinfo (registered above)
            # plus the UINT8 dtype contract. AttnSoftmaxImpl then derives the
            # H*S probability-row qinfo online; do not register H*S static
            # output ranges with AddInt8ConfigPerChannel.
            ctx.Modify(
                {
                    "AddOnPass": [],
                    "DeleteLayer": [],
                    "Layer": [
                        {
                            "layerName": symbol_map[
                                f"layers.{layer_id}.attn_probs"
                            ],
                            "outputDataType": "TFDtypeUint8",
                        }
                        for layer_id in range(config.num_hidden_layers)
                    ],
                }
            )
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
    # Full Quant uses this set as an optimization fence. QuantizeLite does not
    # rewrite the graph, but sharing the metadata keeps both paths auditable.
    source_float_tags: set[str] = {
        "patch.tokens",
        "prefix_concat",
        "add_position_embeddings",
        "tokens.scaled",
        "tokens",
        "final_norm",
        "output.cls",
    }
    for layer_id in range(config.num_hidden_layers):
        prefix = f"layers.{layer_id}"
        source_float_tags.update(
            {
                f"{prefix}.norm1",
                f"{prefix}.proj_tokens.dequantized",
                f"{prefix}.resid1",
                f"{prefix}.norm2",
                f"{prefix}.fc2_tokens.dequantized",
                f"{prefix}.resid2",
            }
        )
        if layer_id in fp_attn_layer_set:
            source_float_tags.update(
                {
                    f"{prefix}.attn.dequantized",
                    f"{prefix}.proj",
                    f"{prefix}.proj_tokens",
                    f"{prefix}.proj.scaled",
                }
            )
        if layer_id in fp_mlp_layer_set:
            source_float_tags.update(
                {
                    f"{prefix}.fc1",
                    f"{prefix}.gate",
                    f"{prefix}.gate_act",
                    f"{prefix}.up",
                    f"{prefix}.mlp_mid",
                    f"{prefix}.fc2",
                    f"{prefix}.fc2_tokens",
                }
            )
    ctx.source_float_tensors = tuple(
        dict.fromkeys(
            symbol_map[tag]
            for tag in source_float_tags
            if tag in symbol_map
        )
    )
    if (
        explicit_qdq
        and fp16_export
        and not attention_scale_requant
        and not per_channel_qk
        and not fold_attention_scale_into_q
    ):
        ctx.source_float_tensors += tuple(
            str(ctx.GetParamSymbol(f"layers.{layer_id}.attn_scale"))
            for layer_id in range(config.num_hidden_layers)
        )
    # These entries are needed only by full Quant, which receives a dtype map
    # in addition to the explicit graph. QuantizeLite uses the source Quantize
    # nodes constructed above instead.
    ctx.quant_entry_tensors = tuple(
        dict.fromkeys(
            [
                symbol_map[f"layers.{layer_id}.norm1"]
                for layer_id in range(config.num_hidden_layers)
            ]
            + [
                symbol_map[f"layers.{layer_id}.norm2"]
                for layer_id in range(config.num_hidden_layers)
                if layer_id not in fp_mlp_layer_set
            ]
        )
    ) if explicit_qdq else ()
    # Full Quant needs the floating side of each DeQuantize as an optimization
    # fence. QuantizeLite must *not* receive this list: although it preserves
    # the source operators, its stop handling inserts another DeQuantize after
    # an already-explicit DQ (mixed-case source name followed by an uppercase
    # generated name).
    ctx.float_entry_tensors = tuple(dict.fromkeys(float_entry_tensors))
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
    convert_float_to_fp16: bool = False,
    quantize_lite: bool = False,
) -> None:
    """Encode a range-annotated graph without changing its precision plan.

    Prefer ``quantize_lite=True`` for source FP16 DeQuantize graphs.  The full
    Quant pass is retained for older FP32 flows because it can discover and
    fold additional scalar operations, but the current SDK rejects FP16 DQ
    destinations during that optimization.  Attention scaling is already a
    source Requant in the Lite flow, so the important fold is not lost.
    """
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
    if quantize_lite:
        # QuantizeLite preserves the graph. Source Quantize operators define
        # every UINT8 entry, so the externally supplied image remains float.
        quant_inputs = {
            name: TFDataType.TFDL_FLOAT for name in tuple(input_names)
        }
    else:
        quant_inputs = {
            name: TFDataType.TFDL_UINT8
            for name in (
                tuple(input_names)
                + tuple(getattr(ctx, "quant_entry_tensors", ()))
            )
        }
    quantize = calib.QuantizeLite if quantize_lite else calib.Quantize
    # MergeConcate=False is intentional. Prefix/register/patch tokens and some
    # experimental per-head tensors can have very different distributions;
    # one merged range saves conversions but is a common source of accuracy
    # loss. Benchmark it explicitly before enabling.
    float_stops = (
        ()
        if quantize_lite
        else tuple(getattr(ctx, "float_entry_tensors", ()))
    )
    quantize(
        quant_inputs,
        stopquanttensors=(
            tuple(output_names)
            + tuple(extra_stopquanttensors)
            + float_stops
        ),
        avoidtensors=tuple(
            dict.fromkeys(
                tuple(avoidtensors)
                + tuple(getattr(ctx, "source_float_tensors", ()))
            )
        ),
        MergeConcate=False,
        Perchannel=True,
    )
    if convert_float_to_fp16:
        # Kept for legacy callers only. New explicit-Q/DQ builds choose FP16
        # parameters and DQ destinations while constructing the source graph.
        calib.ConvertCalibrationFp32ToFp16()
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
    parser.add_argument(
        "--conv-native-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the residual stream as [B,C,N], use LayerNorm2(axis=1), "
            "and let ApplyRope consume Conv-native [B,H,D,N] inputs while "
            "emitting Q=[B,H,N,D], K=[B,H,D,N]. MatMul trans flags remain "
            "disabled for NPU compatibility. Enabled by default; use "
            "--no-conv-native-layout for the legacy [B,N,C] graph."
        ),
    )
    parser.add_argument(
        "--quantize-lite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use QuantizeLite and keep the explicit source Q/DQ topology. "
            "The attention 1/sqrt(head_dim) scalar is emitted as Requantize. "
            "Enabled by default when --dump-quant-fb is requested."
        ),
    )
    parser.add_argument("--dump-symbol-map", default=None)
    parser.add_argument(
        "--outlier-bypass-top-k",
        type=int,
        default=None,
        help=(
            "Globally rank Attention proj and MLP fc2 merge ranges, then "
            "build the largest K branches as explicit floating source islands. "
            "Default: 1 for quantized export, otherwise 0."
        ),
    )
    parser.add_argument(
        "--fp16-export",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Build the floating source islands, residual path, and DeQuantize "
            "destinations as FP16 while retaining LayerNorm parameters in FP32. "
            "Enabled by default when --dump-quant-fb is requested."
        ),
    )
    parser.add_argument(
        "--fp16-residual-scale",
        type=float,
        default=None,
        help=(
            "Positive scale applied from the block-0 MLP residual merge "
            "onward to keep register-token residuals inside FP16. "
            "Default: 0.25 for FP16 models with register tokens, otherwise 1."
        ),
    )
    parser.add_argument(
        "--dump-modify-json",
        default=None,
        help=(
            "Compatibility alias: dump the explicit source Q/DQ precision "
            "plan (no Modify call is generated)."
        ),
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
        "--per-channel-qk",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep all heads in one QK MatMul and register true H*S per-row "
            "ranges for QK output. Requires a range JSON regenerated by the "
            "current Vit.py. QK then uses an identity UINT8 Requantize whose "
            "H*S output qinfo is the QK qinfo divided by sqrt(head_dim), "
            "before UINT8 Softmax. Enabled by default when --dump-quant-fb "
            "is requested."
        ),
    )
    parser.add_argument(
        "--fold-attention-scale-into-q",
        action="store_true",
        help=(
            "Multiply every Q projection weight/bias and Q activation range "
            "by 1/sqrt(head_dim), then connect QK UINT8 directly to Softmax "
            "without Requant/DeQuant/Div. Can be used with scalar or H*S QK."
        ),
    )
    parser.add_argument(
        "--per-row-attention-range-floor",
        type=float,
        default=0.0,
        help=(
            "Expand each QK H*S row range to cover at least this fraction of "
            "its calibrated per-head range. Softmax output keeps scalar "
            "qinfo because AttnSoftmaxImpl quantizes it online."
        ),
    )
    parser.add_argument(
        "--per-channel-qk-max-requant-multiplier",
        type=float,
        default=0.99,
        help=(
            "Maximum Q_scale*K_scale/QK_row_scale allowed for H*S QK. "
            "Rows exceeding this value are expanded about zero while "
            "preserving their min/max ratio. Default: 0.99, keeping the "
            "gemmlowp multiplier strictly below one after float32 rounding."
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


def _resolve_npu_default_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the validated high-accuracy defaults to quantized exports."""
    # Resolve the profile only for model export so calibration-only and
    # external-model comparison commands do not accidentally require a
    # quantized output path. BooleanOptionalAction retains explicit --no-*
    # escape hatches for legacy/ablation graphs.
    quant_export_requested = args.dump_quant_fb is not None
    if args.quantize_lite is None:
        args.quantize_lite = quant_export_requested
    if args.fp16_export is None:
        args.fp16_export = quant_export_requested
    if args.per_channel_qk is None:
        args.per_channel_qk = quant_export_requested
    if args.outlier_bypass_top_k is None:
        args.outlier_bypass_top_k = 1 if quant_export_requested else 0
    return args


def main(argv: list[str] | None = None, default_arch: str | None = None) -> None:
    args = _resolve_npu_default_profile(
        build_arg_parser(default_arch).parse_args(argv)
    )

    if not (0.0 < float(args.register_residual_scale) <= 1.0):
        raise ValueError("--register-residual-scale must be in (0, 1]")
    if float(args.stable_attention_window) < 0.0:
        raise ValueError("--stable-attention-window must be non-negative")
    if args.stable_attention_window and not args.split_attention_heads:
        raise ValueError(
            "--stable-attention-window currently requires --split-attention-heads"
        )
    if args.per_channel_qk and args.split_attention_heads:
        raise ValueError(
            "--per-channel-qk and --split-attention-heads are mutually "
            "exclusive"
        )
    if args.per_channel_qk and not args.quantize_lite:
        raise ValueError("--per-channel-qk currently requires --quantize-lite")
    if not 0.0 <= args.per_row_attention_range_floor <= 1.0:
        raise ValueError("--per-row-attention-range-floor must be in [0, 1]")
    if args.per_channel_qk_max_requant_multiplier is not None and (
        not np.isfinite(args.per_channel_qk_max_requant_multiplier)
        or args.per_channel_qk_max_requant_multiplier <= 0.0
    ):
        raise ValueError(
            "--per-channel-qk-max-requant-multiplier must be positive"
        )
    if args.outlier_bypass_top_k < 0:
        raise ValueError("--outlier-bypass-top-k must be non-negative")
    if (
        args.fp16_residual_scale is not None
        and not (0.0 < float(args.fp16_residual_scale) <= 1.0)
    ):
        raise ValueError("--fp16-residual-scale must be in (0, 1]")
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
            "Top-K source mixed precision is incompatible with "
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
    # An explicitly requested residual reparameterization must also apply to
    # calibration-only runs. Previously --fp16-residual-scale was silently
    # ignored unless --fp16-export was present, so a separately generated
    # range JSON described scale=1 while the later FP16 graph used scale=.25.
    fp16_residual_scale = (
        float(args.fp16_residual_scale)
        if args.fp16_residual_scale is not None
        else (
            0.25
            if args.fp16_export and config.num_register_tokens
            else 1.0
        )
    )
    raw = load_safetensors(args.model_path)
    weights = canonicalize_weights(raw, config)
    apply_fp16_residual_reparameterization(
        weights,
        config,
        fp16_residual_scale,
    )
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
            fp16_residual_scale=fp16_residual_scale,
            conv_native_layout=args.conv_native_layout,
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
                fp16_residual_scale=fp16_residual_scale,
                stable_attention_window=args.stable_attention_window,
            )
            range_json = generated_range_json

        if not needs_graph:
            continue
        explicit_qdq = bool(args.outlier_bypass_top_k or args.fp16_export)
        if args.quantize_lite and not explicit_qdq:
            raise ValueError(
                "--quantize-lite currently requires --fp16-export or "
                "--outlier-bypass-top-k so all INT8/float boundaries are "
                "explicit in the source graph"
            )
        if args.fp16_export and not args.quantize_lite:
            raise ValueError(
                "the current SDK full Quantize pass rejects source "
                "DeQuantize(dstType=FP16); use --quantize-lite, which keeps "
                "the explicit source topology"
            )
        bypass_report = (
            select_outlier_branches(
                config,
                range_json,
                args.outlier_bypass_top_k,
            )
            if explicit_qdq
            else None
        )
        ctx, _, input_names, output_names, symbol_map = build_vit_tfdl_graph(
            config,
            weights,
            range_json=range_json,
            register_residual_scale=args.register_residual_scale,
            fp16_residual_scale=fp16_residual_scale,
            split_mlp_token_groups=args.split_mlp_token_groups,
            split_attention_heads=args.split_attention_heads,
            stable_attention_window=args.stable_attention_window,
            explicit_qdq=explicit_qdq,
            fp_attn_layers=(
                bypass_report["fp_attn_layers"] if bypass_report else ()
            ),
            fp_mlp_layers=(
                bypass_report["fp_mlp_layers"] if bypass_report else ()
            ),
            fp16_export=args.fp16_export,
            attention_scale_requant=(
                args.quantize_lite
                and not args.per_channel_qk
                and not args.fold_attention_scale_into_q
            ),
            source_quantize_entries=args.quantize_lite,
            per_channel_qk=args.per_channel_qk,
            fold_attention_scale_into_q=(
                args.fold_attention_scale_into_q
            ),
            per_row_attention_range_floor=(
                args.per_row_attention_range_floor
            ),
            per_channel_qk_max_requant_multiplier=(
                args.per_channel_qk_max_requant_multiplier
            ),
            conv_native_layout=args.conv_native_layout,
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
            quantize_with_ranges(
                ctx,
                input_names,
                output_names,
                _method_output_path(
                    args.dump_quant_fb, method, multiple_methods
                ),
                extra_stopquanttensors=extra_stopquanttensors,
                convert_float_to_fp16=False,
                quantize_lite=args.quantize_lite,
            )
            quant_output = _method_output_path(
                args.dump_quant_fb, method, multiple_methods
            )
            if quant_output is None:
                raise AssertionError("quant output path was unexpectedly None")
            if bypass_report is not None:
                report = dict(bypass_report)
                report.update(
                    {
                        "graph_rewrite": "source-explicit-qdq",
                        "quantizer": (
                            "TFCalibration.QuantizeLite"
                            if args.quantize_lite
                            else "TFCalibration.Quantize"
                        ),
                        "fp16_export": bool(args.fp16_export),
                        "fp16_residual_scale": float(fp16_residual_scale),
                        "source_dequant_dst_type": (
                            "TFDtypeFp16"
                            if args.fp16_export
                            else "TFDtypeFp32"
                        ),
                        "post_quant_float_conversion": None,
                        "attention_scale": (
                            "Q weight/bias folded; QK directly feeds "
                            "Softmax (scalar or per-channel qinfo)"
                            if args.fold_attention_scale_into_q
                            else "per-channel QK -> identity Requant(UINT8) -> "
                            "H*S score qinfo -> AttnSoftmaxImpl online output "
                            "quantization"
                            if args.per_channel_qk
                            else "source Requantize uint8 lookup table"
                            if args.quantize_lite
                            else "full-Quant scalar optimization"
                        ),
                    }
                )
                report_path = _method_output_path(
                    args.dump_bypass_report_json,
                    method,
                    multiple_methods,
                )
                if report_path:
                    Path(report_path).write_text(
                        json.dumps(report, indent=2, sort_keys=True)
                    )
                source_plan_path = _method_output_path(
                    args.dump_modify_json,
                    method,
                    multiple_methods,
                )
                if source_plan_path:
                    Path(source_plan_path).write_text(
                        json.dumps(report, indent=2, sort_keys=True)
                    )
                selected_summary = ", ".join(
                    f"{item['kind']}:{item['layer']}="
                    f"{item['abs_max']:.6g}"
                    for item in report["selected"]
                )
                print(
                    f"[BYPASS] method={method} top_k="
                    f"{args.outlier_bypass_top_k} fp16={args.fp16_export} "
                    f"residual_scale={fp16_residual_scale:g} "
                    "graph=source-explicit-qdq "
                    f"quantizer={'QuantizeLite' if args.quantize_lite else 'Quantize'} "
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
