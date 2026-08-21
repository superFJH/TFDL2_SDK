#!/usr/bin/env python3
"""Build and calibrate the Mage-VL joint-checkpoint vision tower for TFDL2.

The deployment graph deliberately handles one fixed RGB canvas per invocation.
This is numerically equivalent to the official non-FlashAttention path because
the upstream model splits packed canvases into independent attention sequences.

Inputs, in creation/runtime order:
  0. raw RGB UINT8/FP32 NCHW          [1, 3, H, W]
  1. Mage 3D RoPE sin table FP32      [1, 1, L, 64]
  2. Mage 3D RoPE cos table FP32      [1, 1, L, 64]

Output:
  merged visual embeddings            [1, L/4, 2560]
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

from npu_executor_config import vision_executor_config


VISION_PREFIX = "model.visual."


@dataclass(frozen=True)
class MageVisionConfig:
    canvas_height: int
    canvas_width: int
    patch_size: int
    spatial_merge_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    out_hidden_size: int
    layer_norm_eps: float
    rope_theta: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float

    @property
    def grid_height(self) -> int:
        return self.canvas_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.canvas_width // self.patch_size

    @property
    def seq_len(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def merged_len(self) -> int:
        return self.seq_len // (self.spatial_merge_size**2)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def validate(self) -> None:
        block = self.patch_size * self.spatial_merge_size
        if self.canvas_height % block or self.canvas_width % block:
            raise ValueError(
                "canvas height/width must be divisible by patch_size * spatial_merge_size"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 32:
            raise ValueError("Mage 4:6:6 RoPE requires head_dim divisible by 32")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        canvas_size: tuple[int, int],
        num_layers: int | None = None,
    ) -> "MageVisionConfig":
        root = Path(model_path)
        raw = json.loads((root / "config.json").read_text())
        vision = raw["vision_config"]
        prep_path = root / "preprocessor_config.json"
        prep = json.loads(prep_path.read_text()) if prep_path.exists() else {}
        result = cls(
            canvas_height=int(canvas_size[0]),
            canvas_width=int(canvas_size[1]),
            patch_size=int(vision["patch_size"]),
            spatial_merge_size=int(vision["spatial_merge_size"]),
            hidden_size=int(vision["hidden_size"]),
            intermediate_size=int(vision["intermediate_size"]),
            num_layers=int(
                vision["num_hidden_layers"] if num_layers is None else num_layers
            ),
            num_heads=int(vision["num_attention_heads"]),
            out_hidden_size=int(vision["out_hidden_size"]),
            layer_norm_eps=float(vision.get("layer_norm_eps", 1e-6)),
            rope_theta=float(vision.get("rope_theta", 10000.0)),
            image_mean=tuple(
                float(v)
                for v in prep.get(
                    "image_mean", (0.48145466, 0.4578275, 0.40821073)
                )
            ),
            image_std=tuple(
                float(v)
                for v in prep.get(
                    "image_std", (0.26862954, 0.26130258, 0.27577711)
                )
            ),
            rescale_factor=float(prep.get("rescale_factor", 1.0 / 255.0)),
        )
        result.validate()
        if result.patch_size != 16:
            raise ValueError(
                "Mage-VL joint checkpoint requires patch_size=16; do not use the stale codec patch=14 default"
            )
        return result


def _required_vision_keys(config: MageVisionConfig) -> list[str]:
    keys = [
        VISION_PREFIX + "embeddings.patch_embedding.weight",
        VISION_PREFIX + "layernorm_pre.weight",
        VISION_PREFIX + "layernorm_pre.bias",
        VISION_PREFIX + "merger.ln_q.weight",
        VISION_PREFIX + "merger.ln_q.bias",
        VISION_PREFIX + "merger.mlp.0.weight",
        VISION_PREFIX + "merger.mlp.0.bias",
        VISION_PREFIX + "merger.mlp.2.weight",
        VISION_PREFIX + "merger.mlp.2.bias",
    ]
    for i in range(config.num_layers):
        p = VISION_PREFIX + f"encoder.layers.{i}."
        keys.extend(
            [
                p + "layer_norm1.weight",
                p + "layer_norm1.bias",
                p + "self_attn.qkv.weight",
                p + "self_attn.qkv.bias",
                p + "self_attn.proj.weight",
                p + "self_attn.proj.bias",
                p + "layer_norm2.weight",
                p + "layer_norm2.bias",
                p + "mlp.fc1.weight",
                p + "mlp.fc1.bias",
                p + "mlp.fc2.weight",
                p + "mlp.fc2.bias",
            ]
        )
    return keys


def load_vision_weights(
    model_path: str | Path, config: MageVisionConfig
) -> dict[str, np.ndarray]:
    """Lazy-read only model.visual tensors instead of materializing the 9.5 GB model."""
    from safetensors import safe_open

    root = Path(model_path)
    required = _required_vision_keys(config)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        missing = [key for key in required if key not in weight_map]
        if missing:
            raise KeyError("checkpoint is missing vision keys: " + ", ".join(missing[:8]))
        by_shard: dict[str, list[str]] = {}
        for key in required:
            by_shard.setdefault(weight_map[key], []).append(key)
    else:
        single = root / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError("no safetensors checkpoint/index found")
        by_shard = {single.name: required}

    source: dict[str, np.ndarray] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in keys:
                if key not in available:
                    raise KeyError(f"{key!r} is not present in {shard}")
                source[key] = (
                    handle.get_tensor(key).float().contiguous().numpy()
                )
    return canonicalize_weights(source, config)


def _conv1x1(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError(f"expected [out,in] Linear weight, got {value.shape}")
    return np.ascontiguousarray(value.astype(np.float32)[:, :, None, None])


def canonicalize_weights(
    source: dict[str, np.ndarray], config: MageVisionConfig
) -> dict[str, np.ndarray]:
    def get(suffix: str) -> np.ndarray:
        return np.ascontiguousarray(source[VISION_PREFIX + suffix].astype(np.float32))

    result: dict[str, np.ndarray] = {
        "patch.weight": get("embeddings.patch_embedding.weight"),
        "patch.bias": np.zeros(config.hidden_size, dtype=np.float32),
        "layernorm_pre.weight": get("layernorm_pre.weight"),
        "layernorm_pre.bias": get("layernorm_pre.bias"),
        "merger.norm.weight": get("merger.ln_q.weight"),
        "merger.norm.bias": get("merger.ln_q.bias"),
        "merger.fc1.weight": _conv1x1(get("merger.mlp.0.weight")),
        "merger.fc1.bias": get("merger.mlp.0.bias"),
        "merger.fc2.weight": _conv1x1(get("merger.mlp.2.weight")),
        "merger.fc2.bias": get("merger.mlp.2.bias"),
    }
    for i in range(config.num_layers):
        src = f"encoder.layers.{i}."
        dst = f"layers.{i}."
        result[dst + "norm1.weight"] = get(src + "layer_norm1.weight")
        result[dst + "norm1.bias"] = get(src + "layer_norm1.bias")
        result[dst + "qkv.weight"] = _conv1x1(get(src + "self_attn.qkv.weight"))
        result[dst + "qkv.bias"] = get(src + "self_attn.qkv.bias")
        result[dst + "proj.weight"] = _conv1x1(get(src + "self_attn.proj.weight"))
        result[dst + "proj.bias"] = get(src + "self_attn.proj.bias")
        result[dst + "norm2.weight"] = get(src + "layer_norm2.weight")
        result[dst + "norm2.bias"] = get(src + "layer_norm2.bias")
        result[dst + "fc1.weight"] = _conv1x1(get(src + "mlp.fc1.weight"))
        result[dst + "fc1.bias"] = get(src + "mlp.fc1.bias")
        result[dst + "fc2.weight"] = _conv1x1(get(src + "mlp.fc2.weight"))
        result[dst + "fc2.bias"] = get(src + "mlp.fc2.bias")
    return result


def compute_rope(
    positions: np.ndarray | torch.Tensor,
    theta: float = 10000.0,
    head_dim: int = 64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official Mage sin/cos with shape [1,1,L,head_dim]."""
    positions = torch.as_tensor(positions, dtype=torch.float32, device=device)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("patch_positions must have shape [L,3]")
    if head_dim % 32:
        raise ValueError("head_dim must be divisible by 32")
    half = head_dim // 2
    unit = half // 16

    def inv(size: int) -> torch.Tensor:
        return 1.0 / (
            theta
            ** (
                torch.arange(size, dtype=torch.float32, device=positions.device)
                / size
            )
        )

    freq = torch.cat(
        [
            torch.outer(positions[:, 0], inv(4 * unit)),
            torch.outer(positions[:, 1], inv(6 * unit)),
            torch.outer(positions[:, 2], inv(6 * unit)),
        ],
        dim=-1,
    )
    freq = torch.cat((freq, freq), dim=-1)
    return freq.sin()[None, None], freq.cos()[None, None]


def block_order_positions(config: MageVisionConfig, t: int = 0) -> np.ndarray:
    positions = []
    m = config.spatial_merge_size
    for h0 in range(0, config.grid_height, m):
        for w0 in range(0, config.grid_width, m):
            for dh in range(m):
                for dw in range(m):
                    positions.append((t, h0 + dh, w0 + dw))
    return np.asarray(positions, dtype=np.int64)


def _reorder_patch_tokens(x: torch.Tensor, config: MageVisionConfig) -> torch.Tensor:
    # Conv output is row-major [B,D,H,W]. Qwen2VL/Mage expects 2x2 block order.
    b, d, h, w = x.shape
    m = config.spatial_merge_size
    return (
        x.view(b, d, h // m, m, w // m, m)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(b, h * w, d)
    )


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RangeCollector:
    def __init__(self) -> None:
        self.ranges: dict[str, list[float]] = {}
        # QK is consumed as H*S independent Softmax rows.  Keep the true
        # per-row extrema across calibration canvases so the exported MatMul
        # can use one output qinfo for every (head, query) pair.
        self.row_ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def observe(self, name: str, value: torch.Tensor) -> torch.Tensor:
        data = value.detach().float()
        if name.endswith((".qk_matmul", ".attn_scores")):
            if data.ndim != 4:
                raise ValueError(
                    f"{name} row calibration expects [B,H,S,S], got "
                    f"{tuple(data.shape)}"
                )
            # Aggregate batch and key axes while retaining the runtime H*S
            # order used by AddInt8ConfigPerChannel for MatMul/Softmax.
            row_min = data.amin(dim=(0, 3)).cpu().numpy().reshape(-1)
            row_max = data.amax(dim=(0, 3)).cpu().numpy().reshape(-1)
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
        low = min(0.0, float(data.min().cpu()))
        high = max(0.0, float(data.max().cpu()))
        if name in self.ranges:
            self.ranges[name][0] = min(self.ranges[name][0], low)
            self.ranges[name][1] = max(self.ranges[name][1], high)
        else:
            self.ranges[name] = [low, high]
        return value

    def dump(self, path: str | Path) -> None:
        normalized: dict[str, list[float]] = {}
        for name, values in self.ranges.items():
            low, high = values
            if high - low < 1e-8:
                center = 0.5 * (low + high)
                radius = max(1e-6, abs(center) * 1e-4)
                low, high = center - radius, center + radius
            normalized[name] = [low, high]
        payload = {
            name: {"min": values[0], "max": values[1]}
            for name, values in sorted(normalized.items())
        }
        for name, (row_min, row_max) in sorted(self.row_ranges.items()):
            payload[f"{name}.rows"] = {
                "min": row_min.tolist(),
                "max": row_max.tolist(),
                "range_method": "per-row-minmax",
                "channel_layout": "H*S",
                "row_count": int(row_min.size),
                "calibration_observations": "elementwise union",
            }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


class TorchMageVision:
    """PyTorch graph with the same Linear-as-Conv and token layout as TFDL."""

    def __init__(
        self,
        config: MageVisionConfig,
        weights: dict[str, np.ndarray],
        device: torch.device,
    ) -> None:
        self.config = config
        self.weights = {
            key: torch.from_numpy(value).to(device)
            for key, value in weights.items()
        }
        self.device = device
        self.collector = RangeCollector()

    def w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def linear_conv(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        # x: [B,L,C] -> pointwise Conv -> [B,L,O]
        y = F.conv2d(
            x.transpose(1, 2).unsqueeze(2),
            self.w(prefix + ".weight"),
            self.w(prefix + ".bias"),
        )
        return y.squeeze(2).transpose(1, 2)

    def __call__(
        self, raw_nchw: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        c = self.config
        mean = torch.tensor(c.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(c.image_std, device=self.device).view(1, 3, 1, 1)
        x = (raw_nchw.float() * c.rescale_factor - mean) / std
        x = self.collector.observe("input.normalized", x)
        x = F.conv2d(
            x, self.w("patch.weight"), self.w("patch.bias"),
            stride=c.patch_size,
        )
        x = self.collector.observe("patch.conv", x)
        x = _reorder_patch_tokens(x, c)
        x = self.collector.observe("patch.tokens", x)
        x = F.layer_norm(
            x, (c.hidden_size,), self.w("layernorm_pre.weight"),
            self.w("layernorm_pre.bias"), c.layer_norm_eps,
        )
        x = self.collector.observe("layernorm_pre", x)
        sin, cos = compute_rope(
            positions, c.rope_theta, c.head_dim, device=self.device
        )
        for i in range(c.num_layers):
            p = f"layers.{i}"
            norm = F.layer_norm(
                x, (c.hidden_size,), self.w(p + ".norm1.weight"),
                self.w(p + ".norm1.bias"), c.layer_norm_eps,
            )
            norm = self.collector.observe(p + ".norm1", norm)
            qkv = self.linear_conv(norm, p + ".qkv")
            qkv = self.collector.observe(p + ".qkv", qkv)
            q, k, v = (
                qkv.view(1, c.seq_len, 3, c.num_heads, c.head_dim)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            q = self.collector.observe(p + ".q_pre_rope", q)
            k = self.collector.observe(p + ".k_pre_rope", k)
            v = self.collector.observe(p + ".v", v)
            q = q.float() * cos + _rotate_interleaved(q.float()) * sin
            k = k.float() * cos + _rotate_interleaved(k.float()) * sin
            q = self.collector.observe(p + ".q_rope", q)
            k = self.collector.observe(p + ".k_rope", k)
            qk = torch.matmul(q, k.transpose(-1, -2))
            qk = self.collector.observe(p + ".qk_matmul", qk)
            scores = qk * (c.head_dim**-0.5)
            scores = self.collector.observe(p + ".attn_scores", scores)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32)
            probs = self.collector.observe(p + ".attn_probs", probs)
            attention = torch.matmul(probs, v.float())
            attention = self.collector.observe(p + ".av_matmul", attention)
            attention = (
                attention.transpose(1, 2).contiguous().view(1, c.seq_len, c.hidden_size)
            )
            projected = self.linear_conv(attention, p + ".proj")
            projected = self.collector.observe(p + ".proj", projected)
            x = self.collector.observe(p + ".resid1", x + projected)
            norm2 = F.layer_norm(
                x, (c.hidden_size,), self.w(p + ".norm2.weight"),
                self.w(p + ".norm2.bias"), c.layer_norm_eps,
            )
            norm2 = self.collector.observe(p + ".norm2", norm2)
            mlp = self.linear_conv(norm2, p + ".fc1")
            mlp = self.collector.observe(p + ".fc1", mlp)
            mlp = F.gelu(mlp, approximate="none")
            mlp = self.collector.observe(p + ".gelu", mlp)
            mlp = self.linear_conv(mlp, p + ".fc2")
            mlp = self.collector.observe(p + ".fc2", mlp)
            x = self.collector.observe(p + ".resid2", x + mlp)

        x = F.layer_norm(
            x, (c.hidden_size,), self.w("merger.norm.weight"),
            self.w("merger.norm.bias"), c.layer_norm_eps,
        )
        x = self.collector.observe("merger.norm", x)
        x = x.view(1, c.merged_len, c.hidden_size * c.spatial_merge_size**2)
        x = self.collector.observe("merger.group", x)
        x = self.linear_conv(x, "merger.fc1")
        x = self.collector.observe("merger.fc1", x)
        x = F.gelu(x, approximate="none")
        x = self.collector.observe("merger.gelu", x)
        x = self.linear_conv(x, "merger.fc2")
        return self.collector.observe("output.embeddings", x)


def _mark(symbol_map: dict[str, str], tag: str, symbol: Any) -> Any:
    symbol_map[tag] = str(symbol)
    return symbol


def _pointwise(ctx: Any, x: Any, prefix: str, out_channels: int) -> Any:
    from TFDL2 import Op

    return Op.Convolution2(
        x,
        ctx.GetParamSymbol(prefix + ".weight"),
        ctx.GetParamSymbol(prefix + ".bias"),
        kernel=1,
        pad=0,
        stride=1,
        dilation=1,
        outChannel=out_channels,
        group=1,
    )


def _tokens_to_conv(x: Any, channels: int, length: int) -> Any:
    from TFDL2 import Op

    return Op.Reshape(Op.Transpose(x, (0, 2, 1)), (1, channels, 1, length))


def _conv_to_tokens(x: Any, channels: int, length: int) -> Any:
    from TFDL2 import Op

    return Op.Transpose(Op.Reshape(x, (1, channels, length)), (0, 2, 1))


def _load_row_ranges(
    raw_ranges: dict[str, Any],
) -> dict[str, tuple[list[float], list[float]]]:
    result: dict[str, tuple[list[float], list[float]]] = {}
    for name, item in raw_ranges.items():
        if not isinstance(item, dict):
            continue
        low, high = item.get("min"), item.get("max")
        if not isinstance(low, list) and not isinstance(high, list):
            continue
        if not isinstance(low, list) or not isinstance(high, list):
            raise ValueError(f"{name}: row min/max must both be arrays")
        if not low or len(low) != len(high):
            raise ValueError(
                f"{name}: row min/max lengths must be equal and non-zero"
            )
        row_min = [float(value) for value in low]
        row_max = [float(value) for value in high]
        if not np.all(np.isfinite(row_min)) or not np.all(
            np.isfinite(row_max)
        ):
            raise ValueError(f"{name}: non-finite per-row range")
        if any(lo >= hi for lo, hi in zip(row_min, row_max)):
            raise ValueError(f"{name}: every row range must satisfy min < max")
        result[name] = (row_min, row_max)
    return result


def _linear_requant_maptable(
    input_range: tuple[float, float],
    output_range: tuple[float, float],
    *,
    scale: float,
) -> list[int]:
    """Build the UINT8 lookup used for a scalar linear Requant."""
    input_min, input_max = (float(value) for value in input_range)
    output_min, output_max = (float(value) for value in output_range)
    input_step = (input_max - input_min) / 255.0
    output_step = (output_max - output_min) / 255.0
    if not (
        np.isfinite(input_step)
        and np.isfinite(output_step)
        and np.isfinite(scale)
        and input_step > 0.0
        and output_step > 0.0
    ):
        raise ValueError(
            "invalid scalar Requant parameters: "
            f"input={input_range}, output={output_range}, scale={scale}"
        )
    input_zero = float(np.clip(np.rint(-input_min / input_step), 0.0, 255.0))
    output_zero = float(
        np.clip(np.rint(-output_min / output_step), 0.0, 255.0)
    )
    codes = np.arange(256, dtype=np.float64)
    real = (codes - input_zero) * input_step * float(scale)
    mapped = np.rint(real / output_step + output_zero)
    return np.clip(mapped, 0.0, 255.0).astype(np.uint8).tolist()


def build_tfdl_graph(
    config: MageVisionConfig,
    weights: dict[str, np.ndarray],
    range_json: str | Path | None = None,
    explicit_qdq: bool = False,
    fp_attn_layers: Iterable[int] = (),
    fp_mlp_layers: Iterable[int] = (),
    per_channel_qk: bool = False,
    per_channel_qk_max_requant_multiplier: float = 0.99,
) -> tuple[Any, list[str], list[str], dict[str, str]]:
    from TFDL2 import Op, TFContext
    from TFDL2.Common import TFDataType

    c = config
    qk_max_multiplier = float(per_channel_qk_max_requant_multiplier)
    if not np.isfinite(qk_max_multiplier) or not 0.0 < qk_max_multiplier < 1.0:
        raise ValueError(
            "per-channel QK max requant multiplier must be finite and "
            "strictly between zero and one"
        )
    if per_channel_qk and range_json is None:
        raise ValueError("per-channel QK quantization requires --range-json")
    fp_attn_layer_set = frozenset(int(value) for value in fp_attn_layers)
    fp_mlp_layer_set = frozenset(int(value) for value in fp_mlp_layers)
    raw_ranges = (
        json.loads(Path(range_json).read_text()) if range_json else None
    )
    attention_requant_tables: dict[int, list[int]] = {}
    if explicit_qdq and not per_channel_qk:
        if raw_ranges is None:
            raise ValueError("source attention Requant requires --range-json")
        attention_scale = 1.0 / math.sqrt(c.head_dim)
        for layer_id in range(c.num_layers):
            prefix = f"layers.{layer_id}"
            attention_requant_tables[layer_id] = _linear_requant_maptable(
                _range_item(raw_ranges, f"{prefix}.qk_matmul"),
                _range_item(raw_ranges, f"{prefix}.attn_scores"),
                scale=attention_scale,
            )

    graph_weights = dict(weights)
    if explicit_qdq:
        fp16_params = {
            "merger.fc1.weight",
            "merger.fc1.bias",
            "merger.fc2.weight",
            "merger.fc2.bias",
        }
        for layer_id in fp_attn_layer_set:
            prefix = f"layers.{layer_id}"
            fp16_params.update(
                {f"{prefix}.proj.weight", f"{prefix}.proj.bias"}
            )
        for layer_id in fp_mlp_layer_set:
            prefix = f"layers.{layer_id}"
            for stem in ("fc1", "fc2"):
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
    ctx = TFContext("mage_vit_canvas")
    ctx.RegisterParamToContext(**graph_weights)
    symbols: dict[str, str] = {}
    source_float_symbols: list[str] = []

    def mark_float(tag: str, value: Any) -> Any:
        value = _mark(symbols, tag, value)
        source_float_symbols.append(str(value))
        return value

    def dequantize(tag: str, value: Any) -> Any:
        return mark_float(
            tag,
            Op.DeQuantize(value, TFDataType.TFDL_FLOAT16),
        )

    with ctx:
        input_scale = tuple(
            float(c.rescale_factor / value) for value in c.image_std
        )
        input_mean = tuple(float(value / c.rescale_factor) for value in c.image_mean)
        pixel = Op.Placeholder2(
            ctx,
            shape=(1, 3, c.canvas_height, c.canvas_width),
            outDatatype=TFDataType.TFDL_FLOAT,
            scale=input_scale,
            mean=input_mean,
        )
        pixel = _mark(symbols, "input.normalized", pixel)
        rope_sin = Op.Placeholder2(
            ctx,
            shape=(1, 1, c.seq_len, c.head_dim),
            outDatatype=TFDataType.TFDL_FLOAT,
        )
        rope_cos = Op.Placeholder2(
            ctx,
            shape=(1, 1, c.seq_len, c.head_dim),
            outDatatype=TFDataType.TFDL_FLOAT,
        )
        input_names = [str(pixel), str(rope_sin), str(rope_cos)]

        patch_input = pixel
        if explicit_qdq:
            patch_input = _mark(
                symbols,
                "input.normalized.quantized",
                Op.Quantize(patch_input),
            )
        x = Op.Convolution2(
            patch_input,
            ctx.GetParamSymbol("patch.weight"),
            ctx.GetParamSymbol("patch.bias"),
            kernel=(c.patch_size, c.patch_size),
            pad=0,
            stride=(c.patch_size, c.patch_size),
            dilation=1,
            outChannel=c.hidden_size,
            group=1,
        )
        x = _mark(symbols, "patch.conv", x)
        # [1,D,Hm,M,Wm,M] -> [1,Hm,Wm,M,M,D] -> [1,L,D]
        m = c.spatial_merge_size
        x = Op.Reshape(
            x,
            (1, c.hidden_size, c.grid_height // m, m, c.grid_width // m, m),
        )
        x = Op.Transpose(x, (0, 2, 4, 3, 5, 1))
        x = Op.Reshape(x, (1, c.seq_len, c.hidden_size))
        if explicit_qdq:
            x = _mark(symbols, "patch.tokens.quantized", x)
            x = dequantize("patch.tokens", x)
        else:
            x = _mark(symbols, "patch.tokens", x)
        x = Op.LayerNorm2(
            x,
            ctx.GetParamSymbol("layernorm_pre.weight"),
            ctx.GetParamSymbol("layernorm_pre.bias"),
            axis=-1,
            eps=c.layer_norm_eps,
        )
        x = mark_float("layernorm_pre", x) if explicit_qdq else _mark(
            symbols, "layernorm_pre", x
        )

        for i in range(c.num_layers):
            p = f"layers.{i}"
            norm = Op.LayerNorm2(
                x,
                ctx.GetParamSymbol(p + ".norm1.weight"),
                ctx.GetParamSymbol(p + ".norm1.bias"),
                axis=-1,
                eps=c.layer_norm_eps,
            )
            norm = mark_float(p + ".norm1", norm) if explicit_qdq else _mark(
                symbols, p + ".norm1", norm
            )
            norm_for_qkv = norm
            if explicit_qdq:
                norm_for_qkv = _mark(
                    symbols, p + ".norm1.quantized", Op.Quantize(norm)
                )
            norm1_transposed = _mark(
                symbols,
                p + ".norm1_transposed",
                Op.Transpose(norm_for_qkv, (0, 2, 1)),
            )
            norm1_conv_input = _mark(
                symbols,
                p + ".norm1_conv_input",
                Op.Reshape(
                    norm1_transposed, (1, c.hidden_size, 1, c.seq_len)
                ),
            )
            qkv = _pointwise(
                ctx, norm1_conv_input,
                p + ".qkv", 3 * c.hidden_size,
            )
            qkv = _mark(symbols, p + ".qkv", qkv)
            q4, k4, v4 = Op.Slice(
                qkv, axis=1,
                split=(c.hidden_size, c.hidden_size, c.hidden_size),
            )
            q4 = _mark(symbols, p + ".q_slice", q4)
            k4 = _mark(symbols, p + ".k_slice", k4)
            v4 = _mark(symbols, p + ".v_slice", v4)

            def heads(value: Any, tag: str) -> Any:
                value = Op.Reshape(
                    value, (1, c.num_heads, c.head_dim, c.seq_len)
                )
                value = _mark(symbols, p + f".{tag}_heads_reshape", value)
                return Op.Transpose(value, (0, 1, 3, 2))

            q = _mark(symbols, p + ".q_pre_rope", heads(q4, "q"))
            k = _mark(symbols, p + ".k_pre_rope", heads(k4, "k"))
            v = _mark(symbols, p + ".v", heads(v4, "v"))
            rope = Op.Custom(
                (q, k, rope_sin, rope_cos),
                (f"mage_q_rope_{i}", f"mage_k_rope_{i}"),
                "ApplyRope",
                '{"interleaved":true}',
            )
            q = _mark(symbols, p + ".q_rope", rope[0])
            k = _mark(symbols, p + ".k_rope", rope[1])
            q3 = _mark(
                symbols,
                p + ".q_matmul_input",
                Op.Reshape(q, (c.num_heads, c.seq_len, c.head_dim)),
            )
            k3_reshape = _mark(
                symbols,
                p + ".k_matmul_reshape",
                Op.Reshape(k, (c.num_heads, c.seq_len, c.head_dim)),
            )
            k3 = _mark(
                symbols,
                p + ".k_matmul_input",
                Op.Transpose(k3_reshape, (0, 2, 1)),
            )
            v3 = _mark(
                symbols,
                p + ".v_matmul_input",
                Op.Reshape(v, (c.num_heads, c.seq_len, c.head_dim)),
            )
            qk = _mark(
                symbols, p + ".qk_matmul",
                Op.MatMul(q3, k3, transA=False, transB=False),
            )
            if explicit_qdq:
                # QuantizeLite preserves the source graph. H*S QK uses an
                # identity code map and changes only qinfo; scalar mode uses
                # the equivalent calibrated lookup table.
                scores = Op.Requantize(
                    qk,
                    list(range(256))
                    if per_channel_qk
                    else attention_requant_tables[i],
                )
            else:
                # Full Quantize converts this scalar op into Requant.
                scores = Op.Mul(qk, 1.0 / math.sqrt(c.head_dim))
            scores = _mark(symbols, p + ".attn_scores", scores)
            probs = _mark(
                symbols, p + ".attn_probs", Op.Softmax(scores, axis=2)
            )
            attention = _mark(
                symbols, p + ".av_matmul",
                Op.MatMul(probs, v3, transA=False, transB=False),
            )
            attention = _mark(
                symbols,
                p + ".attn_reshape",
                Op.Reshape(
                attention, (1, c.num_heads, c.seq_len, c.head_dim)
                ),
            )
            attention = _mark(
                symbols,
                p + ".attn_transposed",
                Op.Transpose(attention, (0, 1, 3, 2)),
            )
            attention = _mark(
                symbols,
                p + ".attn_conv_input",
                Op.Reshape(attention, (1, c.hidden_size, 1, c.seq_len)),
            )
            if explicit_qdq and i in fp_attn_layer_set:
                # Top-K Attention keeps QKV/QK/Softmax/AV INT8 and restores
                # only the output projection as one FP16 source island.
                attention = dequantize(
                    p + ".attn_conv_input.dequantized", attention
                )
            projected = _pointwise(
                ctx, attention, p + ".proj", c.hidden_size
            )
            projected = (
                mark_float(p + ".proj_conv", projected)
                if explicit_qdq and i in fp_attn_layer_set
                else _mark(symbols, p + ".proj_conv", projected)
            )
            projected = _conv_to_tokens(projected, c.hidden_size, c.seq_len)
            if explicit_qdq and i not in fp_attn_layer_set:
                projected = _mark(symbols, p + ".proj.quantized", projected)
                projected = dequantize(p + ".proj", projected)
            elif explicit_qdq:
                projected = mark_float(p + ".proj", projected)
            else:
                projected = _mark(symbols, p + ".proj", projected)
            x = Op.Add(x, projected)
            x = mark_float(p + ".resid1", x) if explicit_qdq else _mark(
                symbols, p + ".resid1", x
            )

            norm2 = Op.LayerNorm2(
                x,
                ctx.GetParamSymbol(p + ".norm2.weight"),
                ctx.GetParamSymbol(p + ".norm2.bias"),
                axis=-1,
                eps=c.layer_norm_eps,
            )
            norm2 = mark_float(p + ".norm2", norm2) if explicit_qdq else _mark(
                symbols, p + ".norm2", norm2
            )
            norm2_for_fc1 = norm2
            if explicit_qdq and i not in fp_mlp_layer_set:
                norm2_for_fc1 = _mark(
                    symbols, p + ".norm2.quantized", Op.Quantize(norm2)
                )
            norm2_transposed_value = Op.Transpose(
                norm2_for_fc1, (0, 2, 1)
            )
            norm2_transposed = (
                mark_float(p + ".norm2_transposed", norm2_transposed_value)
                if explicit_qdq and i in fp_mlp_layer_set
                else _mark(
                    symbols,
                    p + ".norm2_transposed",
                    norm2_transposed_value,
                )
            )
            norm2_conv_value = Op.Reshape(
                norm2_transposed, (1, c.hidden_size, 1, c.seq_len)
            )
            norm2_conv_input = (
                mark_float(p + ".norm2_conv_input", norm2_conv_value)
                if explicit_qdq and i in fp_mlp_layer_set
                else _mark(
                    symbols,
                    p + ".norm2_conv_input",
                    norm2_conv_value,
                )
            )
            mlp = _pointwise(
                ctx, norm2_conv_input,
                p + ".fc1", c.intermediate_size,
            )
            mlp = (
                mark_float(p + ".fc1", mlp)
                if explicit_qdq and i in fp_mlp_layer_set
                else _mark(symbols, p + ".fc1", mlp)
            )
            mlp = Op.GeLU(mlp)
            mlp = (
                mark_float(p + ".gelu", mlp)
                if explicit_qdq and i in fp_mlp_layer_set
                else _mark(symbols, p + ".gelu", mlp)
            )
            mlp = _pointwise(ctx, mlp, p + ".fc2", c.hidden_size)
            mlp = (
                mark_float(p + ".fc2_conv", mlp)
                if explicit_qdq and i in fp_mlp_layer_set
                else _mark(symbols, p + ".fc2_conv", mlp)
            )
            mlp = _conv_to_tokens(mlp, c.hidden_size, c.seq_len)
            if explicit_qdq and i not in fp_mlp_layer_set:
                mlp = _mark(symbols, p + ".fc2.quantized", mlp)
                mlp = dequantize(p + ".fc2", mlp)
            elif explicit_qdq:
                mlp = mark_float(p + ".fc2", mlp)
            else:
                mlp = _mark(symbols, p + ".fc2", mlp)
            x = Op.Add(x, mlp)
            x = mark_float(p + ".resid2", x) if explicit_qdq else _mark(
                symbols, p + ".resid2", x
            )

        x = Op.LayerNorm2(
            x,
            ctx.GetParamSymbol("merger.norm.weight"),
            ctx.GetParamSymbol("merger.norm.bias"),
            axis=-1,
            eps=c.layer_norm_eps,
        )
        x = mark_float("merger.norm", x) if explicit_qdq else _mark(
            symbols, "merger.norm", x
        )
        x = Op.Reshape(
            x,
            (1, c.merged_len, c.hidden_size * c.spatial_merge_size**2),
        )
        x = mark_float("merger.group", x) if explicit_qdq else _mark(
            symbols, "merger.group", x
        )
        merger_channels = c.hidden_size * c.spatial_merge_size**2
        merger_group_transposed_value = Op.Transpose(x, (0, 2, 1))
        merger_group_transposed = (
            mark_float("merger.group_transposed", merger_group_transposed_value)
            if explicit_qdq
            else _mark(
                symbols,
                "merger.group_transposed",
                merger_group_transposed_value,
            )
        )
        merger_group_conv_value = Op.Reshape(
            merger_group_transposed,
            (1, merger_channels, 1, c.merged_len),
        )
        merger_group_conv_input = (
            mark_float("merger.group_conv_input", merger_group_conv_value)
            if explicit_qdq
            else _mark(
                symbols,
                "merger.group_conv_input",
                merger_group_conv_value,
            )
        )
        x = _pointwise(
            ctx, merger_group_conv_input,
            "merger.fc1", merger_channels,
        )
        x = mark_float("merger.fc1", x) if explicit_qdq else _mark(
            symbols, "merger.fc1", x
        )
        x = Op.GeLU(x)
        x = mark_float("merger.gelu", x) if explicit_qdq else _mark(
            symbols, "merger.gelu", x
        )
        x = _pointwise(ctx, x, "merger.fc2", c.out_hidden_size)
        x = mark_float("merger.fc2", x) if explicit_qdq else _mark(
            symbols, "merger.fc2", x
        )
        x = _conv_to_tokens(x, c.out_hidden_size, c.merged_len)
        x = mark_float("output.embeddings", x) if explicit_qdq else _mark(
            symbols, "output.embeddings", x
        )
        output_names = [str(x)]
    ctx.SetOutputs(output_names)

    if raw_ranges is not None:
        row_ranges = _load_row_ranges(raw_ranges)
        per_row_configs: dict[str, tuple[list[float], list[float]]] = {}
        if per_channel_qk:
            expected_rows = c.num_heads * c.seq_len
            attention_scale = 1.0 / math.sqrt(c.head_dim)
            for layer_id in range(c.num_layers):
                prefix = f"layers.{layer_id}"
                range_tag = f"{prefix}.qk_matmul.rows"
                if range_tag not in row_ranges:
                    raise KeyError(
                        "--per-channel-qk requires true H*S row ranges; "
                        f"missing {range_tag}. Regenerate the range JSON "
                        "with the current build_mage_vit.py --dump-ranges."
                    )
                qk_min, qk_max = row_ranges[range_tag]
                if len(qk_min) != expected_rows:
                    raise ValueError(
                        f"{range_tag}: got {len(qk_min)} rows, expected "
                        f"H*S={c.num_heads}*{c.seq_len}={expected_rows}"
                    )

                q_min, q_max = _range_item(raw_ranges, f"{prefix}.q_rope")
                k_min, k_max = _range_item(raw_ranges, f"{prefix}.k_rope")
                accumulator_scale = (
                    (q_max - q_min) * (k_max - k_min) / (255.0 * 255.0)
                )
                # gemmlowp's QuantizeMultiplierSmallerThanOne rejects 1.0.
                # nextafter plus a 0.99 default keeps the serialized float32
                # qscale safely on the supported side of that boundary.
                minimum_row_scale = float(
                    np.nextafter(
                        np.float32(accumulator_scale / qk_max_multiplier),
                        np.float32(np.inf),
                    )
                )
                qk_min_array = np.asarray(qk_min, dtype=np.float64)
                qk_max_array = np.asarray(qk_max, dtype=np.float64)
                row_scales = (qk_max_array - qk_min_array) / 255.0
                expand_mask = row_scales < minimum_row_scale
                expanded_rows = int(np.count_nonzero(expand_mask))
                maximum_before = float(np.max(accumulator_scale / row_scales))
                if expanded_rows:
                    factors = np.ones_like(row_scales)
                    factors[expand_mask] = (
                        minimum_row_scale / row_scales[expand_mask]
                    )
                    qk_min_array *= factors
                    qk_max_array *= factors
                qk_min = qk_min_array.astype(np.float32).tolist()
                qk_max = qk_max_array.astype(np.float32).tolist()
                maximum_after = float(
                    np.max(
                        accumulator_scale
                        / ((qk_max_array - qk_min_array) / 255.0)
                    )
                )
                print(
                    f"[MAGE-QK-SCALE-FLOOR] layer={layer_id:02d} "
                    f"expanded={expanded_rows}/{expected_rows} "
                    f"max_multiplier={maximum_before:.6g}->"
                    f"{maximum_after:.6g}"
                )
                qk_name = symbols[f"{prefix}.qk_matmul"]
                score_name = symbols[f"{prefix}.attn_scores"]
                per_row_configs[qk_name] = (qk_max, qk_min)
                per_row_configs[score_name] = (
                    [value * attention_scale for value in qk_max],
                    [value * attention_scale for value in qk_min],
                )

        for tag, symbol in symbols.items():
            if symbol in per_row_configs:
                continue
            range_tag = tag
            for suffix in (".quantized", ".dequantized"):
                if range_tag.endswith(suffix):
                    range_tag = range_tag[: -len(suffix)]
                    break
            for suffix, source_suffix in (
                (".q_matmul_input", ".q_rope"),
                (".k_matmul_input", ".k_rope"),
                (".v_matmul_input", ".v"),
            ):
                if range_tag.endswith(suffix):
                    range_tag = range_tag[: -len(suffix)] + source_suffix
                    break
            for suffix, source_suffix in (
                (".norm1_transposed", ".norm1"),
                (".norm1_conv_input", ".norm1"),
                (".norm2_transposed", ".norm2"),
                (".norm2_conv_input", ".norm2"),
                (".k_matmul_reshape", ".k_rope"),
                (".attn_reshape", ".av_matmul"),
                (".attn_transposed", ".av_matmul"),
                (".attn_conv_input", ".av_matmul"),
                (".proj_conv", ".proj"),
                (".fc2_conv", ".fc2"),
            ):
                if range_tag.endswith(suffix):
                    range_tag = range_tag[: -len(suffix)] + source_suffix
                    break
            if range_tag in {
                "merger.group_transposed",
                "merger.group_conv_input",
            }:
                range_tag = "merger.group"
            if range_tag not in raw_ranges:
                continue
            item = raw_ranges[range_tag]
            low = float(item["min"] if isinstance(item, dict) else item[0])
            high = float(item["max"] if isinstance(item, dict) else item[1])
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(
                    f"invalid range for {tag} (source {range_tag}): "
                    f"[{low}, {high}]"
                )
            if not ctx.AddInt8Config(symbol, high, low):
                raise RuntimeError(f"AddInt8Config failed for {tag} -> {symbol}")
        for symbol, (row_max, row_min) in per_row_configs.items():
            if not ctx.AddInt8ConfigPerChannel(symbol, row_max, row_min):
                raise RuntimeError(
                    "AddInt8ConfigPerChannel failed for "
                    f"{symbol} ({len(row_max)} rows)"
                )
        if explicit_qdq or per_channel_qk:
            # This is the only remaining Modify: AttnSoftmaxImpl quantizes
            # probability rows online and only needs the UINT8 dtype contract.
            ctx.Modify(
                {
                    "AddOnPass": [],
                    "DeleteLayer": [],
                    "Layer": [
                        {
                            "layerName": symbols[
                                f"layers.{layer_id}.attn_probs"
                            ],
                            "outputDataType": "TFDtypeUint8",
                        }
                        for layer_id in range(c.num_layers)
                    ],
                }
            )
    ctx.source_float_tensors = tuple(dict.fromkeys(source_float_symbols))
    return ctx, input_names, output_names, symbols


def _read_ppm(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to read frontend PPM bundles") from exc
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def collect_ranges(
    bundle: str | Path,
    graph: TorchMageVision,
    limit: int = 0,
) -> None:
    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["canvases"]
    if limit > 0:
        entries = entries[:limit]
    for index, entry in enumerate(entries):
        rgb = _read_ppm(root / entry["file"])
        if rgb.shape[:2] != (
            graph.config.canvas_height,
            graph.config.canvas_width,
        ):
            raise ValueError(
                f"canvas {entry['file']} has {rgb.shape[:2]}, expected "
                f"{(graph.config.canvas_height, graph.config.canvas_width)}"
            )
        positions = np.asarray(entry["patch_positions"], dtype=np.int64)
        if positions.shape != (graph.config.seq_len, 3):
            raise ValueError(
                f"canvas {entry['file']} positions have {positions.shape}, "
                f"expected {(graph.config.seq_len, 3)}"
            )
        raw = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(
            graph.device
        )
        pos = torch.from_numpy(positions).to(graph.device)
        with torch.no_grad():
            output = graph(raw, pos)
        print(
            f"[calibration] canvas={index} output={tuple(output.shape)} "
            f"min={float(output.min()):.6g} max={float(output.max()):.6g}"
        )


def _dump_context(ctx: Any, path: str | Path) -> None:
    value = str(path)
    if value.endswith(".fb"):
        value = value[:-3]
    ctx.Dump(value)


def _range_item(ranges: dict[str, Any], tag: str) -> tuple[float, float]:
    if tag not in ranges:
        raise KeyError(f"range JSON is missing {tag!r}")
    item = ranges[tag]
    low = float(item["min"] if isinstance(item, dict) else item[0])
    high = float(item["max"] if isinstance(item, dict) else item[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"invalid range for {tag}: [{low}, {high}]")
    return low, high


def select_outlier_branches(
    config: MageVisionConfig,
    range_json: str | Path,
    top_k: int,
) -> dict[str, Any]:
    """Rank residual-merge Attention/MLP branches by absolute range."""
    if top_k < 0:
        raise ValueError("--outlier-top-k must be non-negative")
    ranges = json.loads(Path(range_json).read_text())
    candidates: list[dict[str, Any]] = []
    for layer_id in range(config.num_layers):
        for kind, suffix in (("attn", "proj"), ("mlp", "fc2")):
            tag = f"layers.{layer_id}.{suffix}"
            low, high = _range_item(ranges, tag)
            candidates.append(
                {
                    "kind": kind,
                    "layer": layer_id,
                    "tag": tag,
                    "min": low,
                    "max": high,
                    "abs_max": max(abs(low), abs(high)),
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


def quantize_graph(
    ctx: Any,
    input_names: list[str],
    output_names: list[str],
    output: str | Path,
    symbols: dict[str, str] | None = None,
    profile: str = "mixed",
    config: MageVisionConfig | None = None,
    weights: dict[str, np.ndarray] | None = None,
    range_json: str | Path | None = None,
    outlier_top_k: int = 0,
    per_channel_qk: bool = False,
    bypass_report: dict[str, Any] | None = None,
    dump_modify_json: str | Path | None = None,
    dump_bypass_report_json: str | Path | None = None,
) -> dict[str, Any] | None:
    from TFDL2 import CalibrationMode, TFCalibration
    from TFDL2.Common import TFDataType

    calibration = TFCalibration(
        ctx,
        CalibrationMode.Naive,
        {
            "UseHardware": False,
            "FrugalMode": True,
            "optimize": {"AttnSoftmaxImpl": True},
        },
    )
    if profile not in {
        "int8-fp16-topk",
        "mixed",
        "all-int8",
    }:
        raise ValueError(f"unknown quantization profile: {profile}")
    if per_channel_qk and profile not in {"int8-fp16-topk", "all-int8"}:
        raise ValueError(
            "per-channel QK currently supports int8-fp16-topk and all-int8; "
            f"profile {profile!r} changes the Softmax path"
        )
    avoid: tuple[str, ...] = ()
    if profile == "mixed":
        if symbols is None:
            raise ValueError("mixed quantization requires the TFDL symbol map")

        def sensitive(tag: str) -> bool:
            return (
                tag == "layernorm_pre"
                or tag.startswith("output.")
                or tag.startswith("merger.norm")
                or tag.startswith("merger.group")
                or tag.endswith(".norm1")
                or tag.endswith(".norm2")
                or tag.endswith(".q_pre_rope")
                or tag.endswith(".k_pre_rope")
                or tag.endswith(".q_rope")
                or tag.endswith(".k_rope")
                or tag.endswith(".v")
                or tag.endswith(".qk_matmul")
                or tag.endswith(".attn_scores")
                or tag.endswith(".attn_probs")
                or tag.endswith(".av_matmul")
                or tag.endswith(".resid1")
                or tag.endswith(".resid2")
            )

        avoid = tuple(
            symbol for tag, symbol in symbols.items() if sensitive(tag)
        )
    if profile == "int8-fp16-topk":
        # The graph already owns every INT8/FP16 boundary. QuantizeLite only
        # encodes weights/qinfo and never rewrites the source topology.
        calibration.QuantizeLite(
            {
                input_names[0]: TFDataType.TFDL_FLOAT,
                input_names[1]: TFDataType.TFDL_FLOAT,
                input_names[2]: TFDataType.TFDL_FLOAT,
            },
            avoidtensors=tuple(
                dict.fromkeys(
                    avoid + tuple(getattr(ctx, "source_float_tensors", ()))
                )
            ),
            stopquanttensors=tuple(output_names),
            MergeConcate=False,
            Perchannel=True,
        )
    else:
        calibration.Quantize(
            {
                input_names[0]: TFDataType.TFDL_UINT8,
                input_names[1]: TFDataType.TFDL_FLOAT,
                input_names[2]: TFDataType.TFDL_FLOAT,
            },
            avoidtensors=avoid,
            stopquanttensors=tuple(output_names),
            MergeConcate=False,
            Perchannel=True,
        )
    ctx.SetOutputs(output_names)
    report = dict(bypass_report) if bypass_report is not None else None
    if report is not None:
        report.update(
            {
                "profile": "int8-fp16-topk",
                "graph_rewrite": "source-explicit-qdq",
                "quantizer": "TFCalibration.QuantizeLite",
                "residual_dtype": "float16",
                "merger_dtype": "float16",
                "attention_scale": (
                    "H*S QK -> identity Requant -> H*S score qinfo"
                    if per_channel_qk
                    else "scalar source Requant lookup"
                ),
                "post_quant_modify": False,
            }
        )
        for path in (dump_modify_json, dump_bypass_report_json):
            if path:
                Path(path).write_text(
                    json.dumps(report, indent=2, sort_keys=True)
                )
    _dump_context(ctx, output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--canvas-size", type=int, nargs=2, default=(288, 512),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help="Bring-up override; omit for the official 24 layers",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=None,
        help=(
            "C++ frontend bundle for calibration; repeat this option to "
            "aggregate ranges from multiple representative videos"
        ),
    )
    parser.add_argument("--max-calib", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dump-ranges", default=None)
    parser.add_argument("--range-json", default=None)
    parser.add_argument("--dump-fb", default=None)
    parser.add_argument("--dump-quant-fb", default=None)
    parser.add_argument(
        "--quant-profile",
        choices=(
            "int8-fp16-topk",
            "mixed",
            "all-int8",
        ),
        default="int8-fp16-topk",
        help=(
            "int8-fp16-topk constructs residual/LN/merger and Top-K bypasses "
            "directly in the source graph, then uses QuantizeLite; mixed and "
            "all-int8 retain the legacy full-Quant comparison paths"
        ),
    )
    parser.add_argument(
        "--outlier-top-k",
        type=int,
        default=2,
        help="Attention-projection/MLP branches restored to FP16 by abs range",
    )
    parser.add_argument(
        "--per-channel-qk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use true H*S QK output qinfo followed by an identity UINT8 "
            "Requant. Enabled by default for quantized exports; use "
            "--no-per-channel-qk only for scalar-QK comparisons."
        ),
    )
    parser.add_argument(
        "--per-channel-qk-max-requant-multiplier",
        type=float,
        default=0.99,
        help=(
            "Expand QK row ranges until Q_scale*K_scale/QK_row_scale is "
            "strictly below one. Default 0.99 leaves float32 safety margin."
        ),
    )
    parser.add_argument(
        "--dump-modify-json",
        default=None,
        help=(
            "Compatibility name: dump the source precision plan/report. "
            "No post-quant graph Modify plan is generated."
        ),
    )
    parser.add_argument("--dump-bypass-report", default=None)
    parser.add_argument("--dump-symbol-map", default=None)
    parser.add_argument(
        "--addon-path",
        default=str(Path(__file__).resolve().parents[3] / "AddonOps/build/libTFDLAddOn.so"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = MageVisionConfig.from_model(
        args.model_path, tuple(args.canvas_size), args.num_layers
    )
    weights = load_vision_weights(args.model_path, config)

    range_json = args.range_json
    if args.dump_ranges:
        if not args.bundle:
            raise ValueError("--dump-ranges requires --bundle")
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        graph = TorchMageVision(config, weights, device)
        for bundle in args.bundle:
            collect_ranges(bundle, graph, args.max_calib)
        graph.collector.dump(args.dump_ranges)
        range_json = args.dump_ranges

    needs_tfdl = bool(args.dump_fb or args.dump_quant_fb or args.dump_symbol_map)
    if not needs_tfdl:
        print(
            f"loaded Mage-ViT: layers={config.num_layers} seq={config.seq_len} "
            f"merged={config.merged_len} output={config.out_hidden_size}"
        )
        return

    addon = Path(args.addon_path)
    if not addon.exists():
        raise FileNotFoundError(f"ApplyRope addon not found: {addon}")
    from TFDL2.utils import LoadCustomOp

    LoadCustomOp(str(addon))
    explicit_qdq = bool(
        args.dump_quant_fb and args.quant_profile == "int8-fp16-topk"
    )
    if explicit_qdq and not range_json:
        raise ValueError(
            "int8-fp16-topk requires --range-json or --dump-ranges"
        )
    bypass_report = (
        select_outlier_branches(config, range_json, args.outlier_top_k)
        if explicit_qdq
        else None
    )
    ctx, inputs, outputs, symbols = build_tfdl_graph(
        config,
        weights,
        range_json=range_json,
        explicit_qdq=explicit_qdq,
        fp_attn_layers=(
            bypass_report["fp_attn_layers"] if bypass_report else ()
        ),
        fp_mlp_layers=(
            bypass_report["fp_mlp_layers"] if bypass_report else ()
        ),
        per_channel_qk=bool(args.per_channel_qk and args.dump_quant_fb),
        per_channel_qk_max_requant_multiplier=(
            args.per_channel_qk_max_requant_multiplier
        ),
    )
    if args.dump_fb:
        _dump_context(ctx, args.dump_fb)
    if args.dump_quant_fb:
        if not range_json:
            raise ValueError(
                "--dump-quant-fb requires --range-json or --dump-ranges"
            )
        quantize_graph(
            ctx, inputs, outputs, args.dump_quant_fb,
            symbols=symbols, profile=args.quant_profile,
            config=config, weights=weights, range_json=range_json,
            outlier_top_k=args.outlier_top_k,
            per_channel_qk=args.per_channel_qk,
            bypass_report=bypass_report,
            dump_modify_json=args.dump_modify_json,
            dump_bypass_report_json=args.dump_bypass_report,
        )
    if args.dump_symbol_map:
        Path(args.dump_symbol_map).write_text(
            json.dumps(symbols, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
