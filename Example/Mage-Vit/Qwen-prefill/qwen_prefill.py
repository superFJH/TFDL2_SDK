#!/usr/bin/env python3
"""Mage-VL Qwen3 prefill primitives for TFDL/NPU experiments.

The prefill contract is deliberately different from the autoregressive decoder
contract used by LocateAnything:

* one fixed-length prompt embedding tensor enters a decoder layer;
* attention is a cache-free, causal native TFDL MatMul/Softmax graph;
* the layer returns its FP16 residual stream and the un-repeated GQA K/V;
* one artifact contains one Qwen layer, keeping compiler peak memory bounded.

The module has no dependency on the existing CPU/GPU ``qwen3_bridge.py``.  It
only shares the prompt embedding/KV semantic contract with that path.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SDK_ROOT = THIS_DIR.parents[2]
PYTHON_SDK_DIR = SDK_ROOT / "Python"
CONVERT_TOOLS_DIR = SDK_ROOT / "ConvertTools" / "python"
MEGAVIT_PYTHON_DIR = THIS_DIR.parent / "python"
PYTHON_BUILD_DIRS = tuple(
    path
    for path in sorted((PYTHON_SDK_DIR / "build").glob("lib.*"))
    if any((path / "TFDL2").glob("TFDL2*.so"))
)
for _path in reversed(
    (*PYTHON_BUILD_DIRS, PYTHON_SDK_DIR, CONVERT_TOOLS_DIR, MEGAVIT_PYTHON_DIR)
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from npu_executor_config import prefill_executor_config  # noqa: E402


@dataclass(frozen=True)
class QwenPrefillConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    attention_bias: bool

    @property
    def kv_repeats(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def query_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    def validate(self) -> None:
        # Qwen3-4B-Instruct-2507 deliberately uses hidden_size=2560 with
        # 32x128=4096 query channels.  Do not infer head_dim from hidden_size.
        if self.hidden_size <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size and head_dim must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise ValueError("Qwen RoPE requires an even head_dim")

    @classmethod
    def from_model(cls, model_path: str | Path) -> "QwenPrefillConfig":
        raw = json.loads((Path(model_path) / "config.json").read_text())
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        num_heads = int(text["num_attention_heads"])
        hidden = int(text["hidden_size"])
        config = cls(
            hidden_size=hidden,
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=int(
                text.get("num_key_value_heads", num_heads)
            ),
            vocab_size=int(text["vocab_size"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(
                text.get("rope_theta", rope.get("rope_theta", 1_000_000.0))
            ),
            head_dim=int(text.get("head_dim", hidden // num_heads)),
            attention_bias=bool(text.get("attention_bias", False)),
        )
        config.validate()
        return config


@dataclass
class TensorRange:
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: Any) -> None:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if not array.size:
            return
        finite = array[np.isfinite(array)]
        if not finite.size:
            return
        self.minimum = min(self.minimum, float(np.min(finite)))
        self.maximum = max(self.maximum, float(np.max(finite)))

    def as_json(self) -> dict[str, float]:
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("cannot serialize an empty tensor range")
        low = self.minimum
        high = self.maximum
        if low == high:
            epsilon = max(abs(low), 1.0) * 1e-6
            low -= epsilon
            high += epsilon
        return {"min": low, "max": high}


class RangeCollector:
    def __init__(self) -> None:
        self.items: dict[str, TensorRange] = {}
        self.row_items: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Projection activations are [B,S,C].  Keeping one range per token
        # makes the fixed-sequence NPU graph equivalent to dynamic per-row
        # activation quantization used by high-quality CPU W8A8 runtimes.
        self.token_items: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _token_rows(name: str, data: np.ndarray) -> np.ndarray | None:
        suffixes_bsc = (
            ".input",
            ".input_norm",
            ".self_attn.q_proj",
            ".self_attn.k_proj",
            ".self_attn.v_proj",
            ".self_attn.o_proj",
            ".attn_residual",
            ".post_norm",
            ".mlp.gate_proj",
            ".mlp.up_proj",
            ".mlp.gate_silu",
            ".mlp.mid",
            ".mlp.down_proj",
            ".output",
        )
        if data.ndim == 3 and name.endswith(suffixes_bsc):
            return data.reshape(-1, data.shape[-1])
        # The eager reference observes attention before heads are folded back
        # into [B,S,H*D].  Transpose it so the exported qinfo still follows S.
        if data.ndim == 4 and name.endswith(".self_attn.attention"):
            return data.transpose(0, 2, 1, 3).reshape(data.shape[0] * data.shape[2], -1)
        return None

    @staticmethod
    def _valid_values(
        name: str, data: np.ndarray, valid_seq_len: int | None
    ) -> np.ndarray:
        """Exclude right-padding and causally masked QK cells from ranges."""
        if valid_seq_len is None:
            return data
        sequence = int(valid_seq_len)
        if sequence <= 0:
            raise ValueError("valid_seq_len must be positive")
        if data.ndim == 3 and data.shape[0] == 1:
            if sequence > data.shape[1]:
                raise ValueError("valid_seq_len exceeds tensor sequence axis")
            return data[:, :sequence]
        if data.ndim == 4 and data.shape[0] == 1:
            if name.endswith((".self_attn.qk_matmul", ".self_attn.scores")):
                if sequence > data.shape[2] or data.shape[2] != data.shape[3]:
                    raise ValueError("invalid QK tensor for padded calibration")
                visible = data[:, :, :sequence, :sequence]
                causal = np.tril(
                    np.ones((sequence, sequence), dtype=bool)
                )
                return visible[:, :, causal]
            # Q/K/V/attention tensors use [B,H,S,D].
            if sequence <= data.shape[2]:
                return data[:, :, :sequence]
        return data

    def observe(
        self, name: str, value: Any, *, valid_seq_len: int | None = None
    ) -> Any:
        data = (
            value.detach().float().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value, dtype=np.float32)
        )
        self.items.setdefault(name, TensorRange()).update(
            self._valid_values(name, data, valid_seq_len)
        )
        token_rows = self._token_rows(name, data)
        if token_rows is not None:
            total_rows = token_rows.shape[0]
            real_rows = total_rows if valid_seq_len is None else int(valid_seq_len)
            if real_rows <= 0 or real_rows > total_rows:
                raise ValueError(f"{name}: invalid real token count {real_rows}")
            token_min = np.full(total_rows, np.inf, dtype=np.float32)
            token_max = np.full(total_rows, -np.inf, dtype=np.float32)
            token_min[:real_rows] = np.minimum(
                token_rows[:real_rows].min(axis=1), 0.0
            )
            token_max[:real_rows] = np.maximum(
                token_rows[:real_rows].max(axis=1), 0.0
            )
            if name in self.token_items:
                previous_min, previous_max = self.token_items[name]
                if previous_min.shape != token_min.shape:
                    raise ValueError(
                        f"{name} token count changed: "
                        f"{previous_min.size} -> {token_min.size}"
                    )
                token_min = np.minimum(previous_min, token_min)
                token_max = np.maximum(previous_max, token_max)
            equal = (token_min == token_max) & np.isfinite(token_min)
            if np.any(equal):
                epsilon = np.maximum(np.abs(token_min[equal]), 1.0) * 1e-6
                token_min[equal] -= epsilon
                token_max[equal] += epsilon
            self.token_items[name] = (token_min, token_max)
        if name.endswith((".self_attn.qk_matmul", ".self_attn.scores")):
            if data.ndim != 4:
                raise ValueError(
                    f"{name} row calibration expects [B,H,S,S], got "
                    f"{tuple(data.shape)}"
                )
            _, heads, sequence, key_sequence = data.shape
            if sequence != key_sequence:
                raise ValueError(f"{name}: QK matrix is not square")
            real_rows = sequence if valid_seq_len is None else int(valid_seq_len)
            if real_rows <= 0 or real_rows > sequence:
                raise ValueError(f"{name}: invalid real query count {real_rows}")
            row_min = np.full((heads, sequence), np.inf, dtype=np.float32)
            row_max = np.full((heads, sequence), -np.inf, dtype=np.float32)
            for query in range(real_rows):
                visible = data[0, :, query, : query + 1]
                row_min[:, query] = np.minimum(visible.min(axis=1), 0.0)
                row_max[:, query] = np.maximum(visible.max(axis=1), 0.0)
            row_min = row_min.reshape(-1)
            row_max = row_max.reshape(-1)
            if name in self.row_items:
                previous_min, previous_max = self.row_items[name]
                if previous_min.shape != row_min.shape:
                    raise ValueError(
                        f"{name} row count changed: "
                        f"{previous_min.size} -> {row_min.size}"
                    )
                row_min = np.minimum(previous_min, row_min)
                row_max = np.maximum(previous_max, row_max)
            equal = (row_min == row_max) & np.isfinite(row_min)
            if np.any(equal):
                epsilon = np.maximum(np.abs(row_min[equal]), 1.0) * 1e-6
                row_min[equal] -= epsilon
                row_max[equal] += epsilon
            self.row_items[name] = (row_min, row_max)
        return value

    def as_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            name: item.as_json()
            for name, item in sorted(self.items.items())
        }
        for name, (row_min, row_max) in sorted(self.row_items.items()):
            if not np.all(np.isfinite(row_min)) or not np.all(np.isfinite(row_max)):
                missing = int(np.count_nonzero(~np.isfinite(row_min)))
                raise ValueError(
                    f"{name} has {missing} uncalibrated HxS rows; include a "
                    "real prompt that reaches the end of the fixed bucket"
                )
            result[name + ".rows"] = {
                "min": row_min.tolist(),
                "max": row_max.tolist(),
                "range_method": "per-row-minmax",
                "channel_layout": "H*S",
                "row_count": int(row_min.size),
                "calibration_observations": "elementwise union",
            }
        for name, (token_min, token_max) in sorted(self.token_items.items()):
            if not np.all(np.isfinite(token_min)) or not np.all(np.isfinite(token_max)):
                missing = int(np.count_nonzero(~np.isfinite(token_min)))
                raise ValueError(
                    f"{name} has {missing} uncalibrated token rows; include a "
                    "real prompt that reaches the end of the fixed bucket"
                )
            result[name + ".tokens"] = {
                "min": token_min.tolist(),
                "max": token_max.tolist(),
                "range_method": "per-token-minmax",
                "channel_layout": "S",
                "row_count": int(token_min.size),
                "calibration_observations": "elementwise union",
            }
        return result

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_json(), indent=2))


class SafeTensorIndex:
    """Small safetensors reader that materializes only requested tensors."""

    def __init__(self, model_path: str | Path) -> None:
        self.root = Path(model_path)
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            self.weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            single = self.root / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"no safetensors checkpoint found below {self.root}"
                )
            with single.open("rb") as handle:
                header_len = int.from_bytes(handle.read(8), "little")
                header = json.loads(handle.read(header_len))
            self.weight_map = {
                name: single.name for name in header if name != "__metadata__"
            }

    def first(self, candidates: Iterable[str]) -> str | None:
        for name in candidates:
            if name in self.weight_map:
                return name
        return None

    @staticmethod
    def _decode(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            value = (u16 << 16).view(np.float32)
        elif dtype == "F16":
            value = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dtype == "F32":
            value = np.frombuffer(raw, dtype="<f4")
        else:
            raise TypeError(f"unsupported weight dtype {dtype}")
        return np.ascontiguousarray(value.reshape(shape))

    def read(self, names: Iterable[str]) -> dict[str, np.ndarray]:
        requested = set(names)
        by_shard: dict[str, list[str]] = {}
        for name in requested:
            shard = self.weight_map.get(name)
            if shard is None:
                raise KeyError(f"checkpoint is missing {name}")
            by_shard.setdefault(shard, []).append(name)
        result: dict[str, np.ndarray] = {}
        for shard, shard_names in sorted(by_shard.items()):
            path = self.root / shard
            with path.open("rb") as handle:
                header_len = int.from_bytes(handle.read(8), "little")
                header = json.loads(handle.read(header_len))
                base = 8 + header_len
                for name in sorted(shard_names):
                    meta = header[name]
                    start, end = (int(x) for x in meta["data_offsets"])
                    handle.seek(base + start)
                    result[name] = self._decode(
                        handle.read(end - start),
                        str(meta["dtype"]),
                        tuple(int(x) for x in meta["shape"]),
                    )
        return result

    def read_rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        """Read selected rows from a 2-D tensor without loading the table."""
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"checkpoint is missing {name}")
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        with (self.root / shard).open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
            meta = header[name]
            shape = tuple(int(x) for x in meta["shape"])
            if len(shape) != 2:
                raise ValueError(f"{name} is not a matrix: {shape}")
            if np.any(rows < 0) or np.any(rows >= shape[0]):
                raise IndexError(f"row outside {name} shape {shape}")
            dtype = str(meta["dtype"])
            item_size = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
            if item_size is None:
                raise TypeError(f"unsupported embedding dtype {dtype}")
            start = int(meta["data_offsets"][0])
            base = 8 + header_len + start
            row_bytes = shape[1] * item_size
            output = np.empty((rows.size, shape[1]), dtype=np.float32)
            for output_row, source_row in enumerate(rows.tolist()):
                handle.seek(base + source_row * row_bytes)
                output[output_row] = self._decode(
                    handle.read(row_bytes), dtype, (shape[1],)
                )
        return output

    def iter_row_blocks(
        self, name: str, block_rows: int = 4096
    ) -> Iterable[tuple[int, np.ndarray]]:
        """Stream a 2-D checkpoint tensor without a full-table allocation."""
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"checkpoint is missing {name}")
        with (self.root / shard).open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
            meta = header[name]
            shape = tuple(int(x) for x in meta["shape"])
            if len(shape) != 2:
                raise ValueError(f"{name} is not a matrix: {shape}")
            dtype = str(meta["dtype"])
            item_size = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
            if item_size is None:
                raise TypeError(f"unsupported matrix dtype {dtype}")
            data_start = (
                8 + header_len + int(meta["data_offsets"][0])
            )
            row_bytes = shape[1] * item_size
            for row in range(0, shape[0], block_rows):
                count = min(block_rows, shape[0] - row)
                handle.seek(data_start + row * row_bytes)
                yield row, self._decode(
                    handle.read(count * row_bytes),
                    dtype,
                    (count, shape[1]),
                )


def _layer_source_candidates(layer_id: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.language_model.layers.{layer_id}.{suffix}",
        f"language_model.model.layers.{layer_id}.{suffix}",
        f"model.layers.{layer_id}.{suffix}",
    )


LAYER_WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def load_layer_weights(
    model_path: str | Path,
    layer_id: int,
    index: SafeTensorIndex | None = None,
) -> dict[str, np.ndarray]:
    index = index or SafeTensorIndex(model_path)
    source_to_target: dict[str, str] = {}
    for suffix in LAYER_WEIGHT_SUFFIXES:
        source = index.first(_layer_source_candidates(layer_id, suffix))
        if source is None:
            raise KeyError(
                f"layer {layer_id} is missing {suffix}; tried Mage/Qwen prefixes"
            )
        source_to_target[source] = f"layers.{layer_id}.{suffix}"
    source_values = index.read(source_to_target)
    return {
        source_to_target[source]: value
        for source, value in source_values.items()
    }


def embedding_weight_name(index: SafeTensorIndex) -> str:
    name = index.first(
        (
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
    )
    if name is None:
        raise KeyError("checkpoint has no Qwen token embedding table")
    return name


def final_weight_names(index: SafeTensorIndex) -> tuple[str, str]:
    norm = index.first(
        (
            "model.language_model.norm.weight",
            "language_model.model.norm.weight",
            "model.norm.weight",
        )
    )
    head = index.first(
        (
            "lm_head.weight",
            "model.language_model.lm_head.weight",
            "language_model.lm_head.weight",
        )
    )
    if norm is None or head is None:
        raise KeyError("checkpoint is missing final norm or lm_head")
    return norm, head


def compute_final_logits(
    model_path: str | Path,
    hidden: np.ndarray,
    config: QwenPrefillConfig,
    *,
    device: str = "cpu",
    block_rows: int = 4096,
    index: SafeTensorIndex | None = None,
) -> np.ndarray:
    """Compute final RMSNorm + LM head for the last prompt token.

    The LM head is streamed in row blocks, so CPU prefill evaluation does not
    temporarily expand the roughly 0.8 GiB BF16 matrix to a 1.6 GiB FP32
    allocation.  A non-CPU device uses the same streaming contract and moves
    one block at a time.
    """
    index = index or SafeTensorIndex(model_path)
    norm_name, head_name = final_weight_names(index)
    norm_weight = index.read((norm_name,))[norm_name].reshape(-1)
    value = np.asarray(hidden, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            f"hidden must have shape [1,S,D], got {value.shape}"
        )
    last = value[0, -1]
    if last.size != config.hidden_size or norm_weight.size != last.size:
        raise ValueError("final hidden/norm width does not match config")
    normalized = last / np.sqrt(
        np.mean(last.astype(np.float64) ** 2) + config.rms_norm_eps
    )
    normalized = np.ascontiguousarray(
        normalized * norm_weight, dtype=np.float32
    )
    logits = np.empty(config.vocab_size, dtype=np.float32)
    if device == "cpu":
        for begin, weight in index.iter_row_blocks(head_name, block_rows):
            end = begin + weight.shape[0]
            logits[begin:end] = weight @ normalized
        return logits

    import torch

    target = torch.device(device)
    hidden_tensor = torch.from_numpy(normalized).to(target)
    with torch.inference_mode():
        for begin, weight in index.iter_row_blocks(head_name, block_rows):
            end = begin + weight.shape[0]
            block = torch.from_numpy(weight).to(target)
            logits[begin:end] = (
                torch.mv(block, hidden_tensor).float().cpu().numpy()
            )
    return logits


def compute_rope(
    position_ids: np.ndarray,
    head_dim: int,
    rope_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(position_ids, dtype=np.float32).reshape(-1)
    inv_freq = 1.0 / (
        rope_theta
        ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    frequencies = np.einsum("s,d->sd", positions, inv_freq)
    angles = np.concatenate((frequencies, frequencies), axis=-1)
    sin = np.sin(angles).reshape(1, 1, positions.size, head_dim)
    cos = np.cos(angles).reshape(1, 1, positions.size, head_dim)
    return np.ascontiguousarray(sin, dtype=np.float32), np.ascontiguousarray(
        cos, dtype=np.float32
    )


def causal_mask(seq_len: int, dtype: np.dtype = np.float32) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    # -65504 is finite in FP16 and behaves as -inf after Softmax.
    mask = np.zeros((1, seq_len, seq_len), dtype=np.float32)
    mask[0][np.triu_indices(seq_len, k=1)] = -65504.0
    return np.ascontiguousarray(mask.astype(dtype))


def infer_token_group_boundaries(
    prompt_dir: str | Path, seq_len: int
) -> tuple[int, ...]:
    """Infer prefix/visual-context/final-query spans from a prepared prompt."""
    root = Path(prompt_dir)
    metadata_path = root / "metadata.json"
    input_ids_path = root / "input_ids.npy"
    if not metadata_path.exists() or not input_ids_path.exists():
        raise FileNotFoundError(
            "Tok hybrid inference requires metadata.json and input_ids.npy "
            f"below {root}"
        )
    metadata = json.loads(metadata_path.read_text())
    if "image_token_id" not in metadata:
        raise KeyError(f"{metadata_path} has no image_token_id")
    input_ids = np.load(input_ids_path).reshape(-1)
    if input_ids.size != seq_len:
        raise ValueError(
            f"prompt has {input_ids.size} tokens, expected seq_len={seq_len}"
        )
    valid_seq_len = int(metadata.get("valid_seq_len", seq_len))
    return prompt_token_group_boundaries(
        input_ids,
        int(metadata["image_token_id"]),
        valid_seq_len=valid_seq_len,
    )


def prompt_token_group_boundaries(
    input_ids: np.ndarray,
    image_token_id: int,
    *,
    valid_seq_len: int | None = None,
) -> tuple[int, ...]:
    """Return the actual prefix/vision/query boundaries of one prompt.

    Tok-hybrid FBs bake these positions into Slice/Concat shapes. Computing
    the same values at runtime lets a caller detect a video/template whose
    framing tokens put the final query in a different quantization group.
    Right-padding is deliberately excluded from the scan.
    """
    ids = np.asarray(input_ids).reshape(-1)
    valid = ids.size if valid_seq_len is None else int(valid_seq_len)
    if not 0 < valid <= ids.size:
        raise ValueError(
            f"valid_seq_len={valid}, expected inside [1, {ids.size}]"
        )
    visual = np.flatnonzero(ids[:valid] == int(image_token_id))
    if not visual.size:
        raise ValueError("prompt contains no image tokens for Tok hybrid")
    candidates = (int(visual[0]), int(visual[-1]) + 1)
    return tuple(value for value in candidates if 0 < value < ids.size)


def cosine(reference: np.ndarray, actual: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(actual, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _torch_rms_norm(value: Any, weight: Any, eps: float) -> Any:
    import torch

    variance = value.float().square().mean(dim=-1, keepdim=True)
    return value * torch.rsqrt(variance + eps).to(value.dtype) * weight.to(
        value.device, value.dtype
    )


def _torch_apply_rope(value: Any, sin: Any, cos: Any) -> Any:
    import torch

    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
    return value * cos + rotated * sin


def torch_layer(
    config: QwenPrefillConfig,
    layer_id: int,
    weights: dict[str, np.ndarray],
    hidden: Any,
    sin: Any,
    cos: Any,
    collector: RangeCollector | None = None,
    valid_seq_len: int | None = None,
) -> tuple[Any, Any, Any]:
    """Exact eager Qwen3 layer used for range collection and parity tests."""
    import torch
    import torch.nn.functional as functional

    prefix = f"layers.{layer_id}"

    def weight(suffix: str) -> Any:
        return torch.as_tensor(
            weights[f"{prefix}.{suffix}"],
            dtype=hidden.dtype,
            device=hidden.device,
        )

    def observe(tag: str, value: Any) -> Any:
        if collector is not None:
            collector.observe(
                f"layers.{layer_id}.{tag}",
                value,
                valid_seq_len=valid_seq_len,
            )
        return value

    hidden = observe("input", hidden)
    normalized = observe(
        "input_norm",
        _torch_rms_norm(
            hidden, weight("input_layernorm.weight"), config.rms_norm_eps
        ),
    )
    q = observe(
        "self_attn.q_proj", functional.linear(normalized, weight("self_attn.q_proj.weight"))
    )
    k = observe(
        "self_attn.k_proj", functional.linear(normalized, weight("self_attn.k_proj.weight"))
    )
    v = observe(
        "self_attn.v_proj", functional.linear(normalized, weight("self_attn.v_proj.weight"))
    )
    batch, seq_len, _ = hidden.shape
    q = q.reshape(batch, seq_len, config.num_attention_heads, config.head_dim)
    k = k.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    q = observe(
        "self_attn.q_norm",
        _torch_rms_norm(q, weight("self_attn.q_norm.weight"), config.rms_norm_eps),
    ).transpose(1, 2)
    k = observe(
        "self_attn.k_norm",
        _torch_rms_norm(k, weight("self_attn.k_norm.weight"), config.rms_norm_eps),
    ).transpose(1, 2)
    v = v.reshape(
        batch, seq_len, config.num_key_value_heads, config.head_dim
    ).transpose(1, 2)
    q = observe("self_attn.q_rope", _torch_apply_rope(q, sin, cos))
    k = observe("self_attn.k_rope", _torch_apply_rope(k, sin, cos))
    v = observe("self_attn.v_cache", v)
    repeated_k = k.repeat_interleave(config.kv_repeats, dim=1)
    repeated_v = v.repeat_interleave(config.kv_repeats, dim=1)
    qk = observe(
        "self_attn.qk_matmul",
        torch.matmul(q, repeated_k.transpose(-1, -2)),
    )
    scores = observe(
        "self_attn.scores", qk * (config.head_dim**-0.5)
    )
    mask = torch.full(
        (seq_len, seq_len),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device,
    ).triu(1)
    scores = observe("self_attn.masked_scores", scores + mask)
    probabilities = observe("self_attn.probabilities", torch.softmax(scores, dim=-1))
    attention = observe("self_attn.attention", torch.matmul(probabilities, repeated_v))
    attention = attention.transpose(1, 2).reshape(batch, seq_len, -1)
    projected = observe(
        "self_attn.o_proj",
        functional.linear(attention, weight("self_attn.o_proj.weight")),
    )
    residual = observe("attn_residual", hidden + projected)
    post_norm = observe(
        "post_norm",
        _torch_rms_norm(
            residual,
            weight("post_attention_layernorm.weight"),
            config.rms_norm_eps,
        ),
    )
    gate = observe(
        "mlp.gate_proj", functional.linear(post_norm, weight("mlp.gate_proj.weight"))
    )
    up = observe(
        "mlp.up_proj", functional.linear(post_norm, weight("mlp.up_proj.weight"))
    )
    gate_silu = observe("mlp.gate_silu", functional.silu(gate))
    mlp_mid = observe("mlp.mid", gate_silu * up)
    down = observe(
        "mlp.down_proj", functional.linear(mlp_mid, weight("mlp.down_proj.weight"))
    )
    output = observe("output", residual + down)
    return output, k, v


def _load_tfdl(addon_path: str | Path | None = None) -> tuple[Any, ...]:
    from TFDL2 import Op, TFContext, TFExecutor
    from TFDL2.Common import TFDataType
    from TFDL2.utils import LoadCustomOp

    addon = Path(addon_path) if addon_path else (
        SDK_ROOT / "AddonOps" / "build" / "libTFDLAddOn.so"
    )
    if not addon.exists():
        raise FileNotFoundError(f"TFDL addon is missing: {addon}")
    LoadCustomOp(str(addon))
    return Op, TFContext, TFExecutor, TFDataType


def _conv_alias(name: str) -> str:
    return f"__conv1x1__{name}"


def _matmul_alias(name: str) -> str:
    return f"__matmul__{name}"


def _factor_token_grid(token_count: int) -> tuple[int, int]:
    """Return the least-skewed exact HxW mapping for a token axis."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    height = math.isqrt(token_count)
    while token_count % height:
        height -= 1
    return height, token_count // height


def _prepare_params(
    weights: dict[str, np.ndarray], dtype: np.dtype = np.float32
) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    for name, value in weights.items():
        array = np.ascontiguousarray(value, dtype=dtype)
        if array.ndim == 2:
            params[_conv_alias(name)] = array.reshape(
                array.shape[0], array.shape[1], 1, 1
            )
        else:
            params[name] = array
    return params


def _linear_conv(
    Op: Any,
    ctx: Any,
    value: Any,
    weight_name: str,
    in_channels: int,
    out_channels: int,
    seq_len: int,
) -> tuple[Any, Any, Any, Any]:
    grid_height, grid_width = _factor_token_grid(seq_len)
    transposed = Op.Transpose(value, (0, 2, 1))
    conv_input = Op.Reshape(
        transposed, (1, in_channels, grid_height, grid_width)
    )
    output = Op.Convolution2(
        conv_input,
        ctx.GetParamSymbol(_conv_alias(weight_name)),
        None,
        kernel=1,
        pad=0,
        stride=1,
        dilation=1,
        outChannel=out_channels,
        group=1,
    )
    tokens = Op.Transpose(
        Op.Reshape(output, (1, out_channels, seq_len)), (0, 2, 1)
    )
    return tokens, transposed, conv_input, output


def _rms_norm(
    Op: Any,
    ctx: Any,
    value: Any,
    weight_name: str,
    eps: float,
    output_name: str,
) -> tuple[Any, Any]:
    raw = Op.Custom(
        (value,),
        (output_name + "_raw",),
        "RMSNorm",
        json.dumps({"eps": float(eps)}),
    )
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return Op.Mul(raw, ctx.GetParamSymbol(weight_name)), raw


def _repeat_gqa(Op: Any, value: Any, config: QwenPrefillConfig) -> Any:
    if config.kv_repeats == 1:
        return value
    heads = Op.Slice(
        value,
        axis=1,
        split=tuple(1 for _ in range(config.num_key_value_heads)),
    )
    repeated = tuple(
        head
        for head in heads
        for _ in range(config.kv_repeats)
    )
    return Op.Concat(repeated, axis=1)


def build_layer_graph(
    config: QwenPrefillConfig,
    layer_id: int,
    seq_len: int,
    weights: dict[str, np.ndarray],
    *,
    range_json: str | Path | None = None,
    addon_path: str | Path | None = None,
    create_executor: bool = False,
    use_hardware: bool = False,
    causal: bool = True,
    qk_norm: bool = True,
    export_kv: bool = True,
    fp16_boundaries: bool = False,
) -> tuple[Any, Any | None, list[str], list[str], dict[str, str]]:
    """Build one exact, cache-free Qwen3 prefill layer."""
    if not 0 <= layer_id < config.num_hidden_layers:
        raise ValueError(f"layer_id outside [0,{config.num_hidden_layers})")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    Op, TFContext, TFExecutor, TFDataType = _load_tfdl(addon_path)
    prefix = f"layers.{layer_id}"
    ctx = TFContext(f"MageQwenPrefillLayer{layer_id}Seq{seq_len}")
    params = _prepare_params(
        weights, np.float16 if fp16_boundaries else np.float32
    )
    attention_scale_name = f"__qwen_prefill_attention_scale_{layer_id}"
    if fp16_boundaries:
        params[attention_scale_name] = np.asarray(
            [config.head_dim**-0.5], dtype=np.float16
        )
    ctx.RegisterParamToContext(**params)
    available = set(params)
    del params
    gc.collect()
    symbols: dict[str, str] = {}

    def mark(tag: str, value: Any) -> Any:
        symbols[f"{prefix}.{tag}"] = str(value)
        return value

    def conv(
        value: Any, suffix: str, in_channels: int, out_channels: int
    ) -> Any:
        weight_name = f"{prefix}.{suffix}.weight"
        if _conv_alias(weight_name) not in available:
            raise KeyError(f"graph is missing {weight_name}")
        output, transposed, conv_input, conv_output = _linear_conv(
            Op,
            ctx,
            value,
            weight_name,
            in_channels,
            out_channels,
            seq_len,
        )
        mark(suffix + ".transposed", transposed)
        mark(suffix + ".conv_input", conv_input)
        mark(suffix + ".conv", conv_output)
        return mark(suffix, output)

    with ctx:
        boundary_dtype = (
            TFDataType.TFDL_FLOAT16
            if fp16_boundaries
            else TFDataType.TFDL_FLOAT
        )
        hidden = Op.Placeholder2(
            ctx, (1, seq_len, config.hidden_size), boundary_dtype
        )
        rope_sin = Op.Placeholder2(
            ctx, (1, 1, seq_len, config.head_dim), TFDataType.TFDL_FLOAT
        )
        rope_cos = Op.Placeholder2(
            ctx, (1, 1, seq_len, config.head_dim), TFDataType.TFDL_FLOAT
        )
        symbols[f"{prefix}.input"] = str(hidden)
        symbols["rope_sin"] = str(rope_sin)
        symbols["rope_cos"] = str(rope_cos)
        input_names = [str(hidden), str(rope_sin), str(rope_cos)]
        mask = None
        if causal:
            mask = Op.Placeholder2(ctx, (1, seq_len, seq_len), boundary_dtype)
            symbols["causal_mask"] = str(mask)
            input_names.append(str(mask))

        input_norm, input_rms = _rms_norm(
            Op,
            ctx,
            hidden,
            f"{prefix}.input_layernorm.weight",
            config.rms_norm_eps,
            f"qwen_prefill_l{layer_id}_input_rms",
        )
        mark("input_rms", input_rms)
        input_norm = mark("input_norm", input_norm)

        q = conv(
            input_norm,
            "self_attn.q_proj",
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        )
        k = conv(
            input_norm,
            "self_attn.k_proj",
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )
        v = conv(
            input_norm,
            "self_attn.v_proj",
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )
        q = mark("self_attn.q_heads_reshape", Op.Reshape(
            q,
            (1, seq_len, config.num_attention_heads, config.head_dim),
        ))
        k = mark("self_attn.k_heads_reshape", Op.Reshape(
            k,
            (1, seq_len, config.num_key_value_heads, config.head_dim),
        ))
        if qk_norm:
            q, q_rms = _rms_norm(
                Op,
                ctx,
                q,
                f"{prefix}.self_attn.q_norm.weight",
                config.rms_norm_eps,
                f"qwen_prefill_l{layer_id}_q_rms",
            )
            k, k_rms = _rms_norm(
                Op,
                ctx,
                k,
                f"{prefix}.self_attn.k_norm.weight",
                config.rms_norm_eps,
                f"qwen_prefill_l{layer_id}_k_rms",
            )
            mark("self_attn.q_rms", q_rms)
            mark("self_attn.k_rms", k_rms)
        q = mark("self_attn.q_norm", q)
        k = mark("self_attn.k_norm", k)
        q = mark("self_attn.q_heads", Op.Transpose(q, (0, 2, 1, 3)))
        k = mark("self_attn.k_heads", Op.Transpose(k, (0, 2, 1, 3)))
        v = mark("self_attn.v_heads", Op.Transpose(
            mark("self_attn.v_heads_reshape", Op.Reshape(
                v,
                (1, seq_len, config.num_key_value_heads, config.head_dim),
            )),
            (0, 2, 1, 3),
        ))
        rope = Op.Custom(
            (q, k, rope_sin, rope_cos),
            (
                f"qwen_prefill_l{layer_id}_q_rope",
                f"qwen_prefill_l{layer_id}_k_rope",
            ),
            "ApplyRope",
            "{}",
        )
        q = mark("self_attn.q_rope", rope[0])
        k = mark("self_attn.k_rope", rope[1])
        v = mark("self_attn.v_cache", v)

        repeated_k = mark("self_attn.k_repeated", _repeat_gqa(Op, k, config))
        repeated_v = mark("self_attn.v_repeated", _repeat_gqa(Op, v, config))
        q3 = Op.Reshape(
            q, (config.num_attention_heads, seq_len, config.head_dim)
        )
        k3 = Op.Reshape(
            repeated_k,
            (config.num_attention_heads, seq_len, config.head_dim),
        )
        k3 = Op.Transpose(k3, (0, 2, 1))
        v3 = Op.Reshape(
            repeated_v,
            (config.num_attention_heads, seq_len, config.head_dim),
        )
        mark("self_attn.q_matmul_input", q3)
        mark("self_attn.k_matmul_input", k3)
        mark("self_attn.v_matmul_input", v3)
        scores = mark(
            "self_attn.qk_matmul",
            Op.MatMul(q3, k3, transA=False, transB=False),
        )
        scores = mark(
            "self_attn.scores",
            Op.Mul(
                scores,
                ctx.GetParamSymbol(attention_scale_name)
                if fp16_boundaries
                else config.head_dim**-0.5,
            ),
        )
        if mask is not None:
            scores = mark("self_attn.masked_scores", Op.Add(scores, mask))
        else:
            mark("self_attn.masked_scores", scores)
        probabilities = mark(
            "self_attn.probabilities", Op.Softmax(scores, axis=2)
        )
        attention = mark(
            "self_attn.attention",
            Op.MatMul(probabilities, v3, transA=False, transB=False),
        )
        attention = mark("self_attn.attention_reshape", Op.Reshape(
            attention,
            (1, config.num_attention_heads, seq_len, config.head_dim),
        ))
        attention = mark(
            "self_attn.attention_transposed",
            Op.Transpose(attention, (0, 2, 1, 3)),
        )
        attention = mark(
            "self_attn.attention_tokens",
            Op.Reshape(attention, (1, seq_len, config.query_size)),
        )
        projected = conv(
            attention,
            "self_attn.o_proj",
            config.query_size,
            config.hidden_size,
        )
        residual = mark("attn_residual", Op.Add(hidden, projected))

        post_norm, post_rms = _rms_norm(
            Op,
            ctx,
            residual,
            f"{prefix}.post_attention_layernorm.weight",
            config.rms_norm_eps,
            f"qwen_prefill_l{layer_id}_post_rms",
        )
        mark("post_rms", post_rms)
        post_norm = mark("post_norm", post_norm)
        gate = conv(
            post_norm,
            "mlp.gate_proj",
            config.hidden_size,
            config.intermediate_size,
        )
        up = conv(
            post_norm,
            "mlp.up_proj",
            config.hidden_size,
            config.intermediate_size,
        )
        gate = mark("mlp.gate_silu", Op.Swish(gate))
        mlp_mid = mark("mlp.mid", Op.Mul(gate, up))
        down = conv(
            mlp_mid,
            "mlp.down_proj",
            config.intermediate_size,
            config.hidden_size,
        )
        output = mark("output", Op.Add(residual, down))
        output_names = [str(output)]
        if export_kv:
            cache_shape = (
                1,
                config.num_key_value_heads,
                seq_len,
                config.head_dim,
            )
            # TFDL's dump optimizer can omit a directly exported custom-op
            # output when it is also consumed by attention. Zero-compute
            # terminal Reshapes preserve the three-output ABI.
            k_export = mark(
                "self_attn.k_export", Op.Reshape(k, cache_shape)
            )
            v_export = mark(
                "self_attn.v_export", Op.Reshape(v, cache_shape)
            )
            output_names.extend((str(k_export), str(v_export)))

    ctx.SetOutputs(output_names)
    if range_json:
        ranges = json.loads(Path(range_json).read_text())
        for logical_name, symbol_name in symbols.items():
            item = ranges.get(logical_name)
            if item is None:
                continue
            low = float(item["min"] if isinstance(item, dict) else item[0])
            high = float(item["max"] if isinstance(item, dict) else item[1])
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(
                    f"invalid range for {logical_name}: [{low}, {high}]"
                )
            if not ctx.AddInt8Config(symbol_name, high, low):
                raise RuntimeError(
                    f"AddInt8Config failed for {logical_name} -> {symbol_name}"
                )
    executor = None
    if create_executor:
        executor = TFExecutor(
            ctx,
            prefill_executor_config(
                bool(use_hardware),
                software_attn_softmax_impl=True,
            ),
        )
    return ctx, executor, input_names, output_names, symbols


def dump_context(ctx: Any, output: str | Path) -> None:
    path = str(output)
    if path.endswith(".fb"):
        path = path[:-3]
    ctx.Dump(path)


def audit_exported_int8_qinfo(
    fb_path: str | Path,
    addon_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect every exported tensor and validate UINT8 quantization data.

    The logical symbol map is intentionally not used here. Quantization or
    layout passes can create unnamed intermediate Reshape/Transpose tensors;
    ``GetAllTensorNames`` is the only reliable final-FB inventory.
    """
    _, TFContext, TFExecutor, _ = _load_tfdl(addon_path)
    path = Path(fb_path).resolve()
    context = TFContext(path=str(path))
    executor = TFExecutor(
        context,
        {
            "UseHardware": False,
            "FrugalMode": False,
            "optimize": {"AttnSoftmaxImpl": False},
        },
    )
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    for name in sorted(context.GetAllTensorNames()):
        row: dict[str, Any] = {"name": name}
        try:
            tensor = executor.GetTensorByName(name)
            dtype = str(tensor.dtype)
            qmin = np.asarray(tensor.qmin, dtype=np.float64)
            qmax = np.asarray(tensor.qmax, dtype=np.float64)
            qscale = np.asarray(tensor.qscale, dtype=np.float64)
            qzero = np.asarray(tensor.qzeropoint, dtype=np.float64)
            row.update(
                {
                    "dtype": dtype,
                    "shape": [int(value) for value in tensor.shape],
                    "qinfo_count": int(qscale.size),
                }
            )
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            reasons: list[str] = []
            if "UINT8" in dtype:
                counts = (qmin.size, qmax.size, qscale.size, qzero.size)
                if not all(counts):
                    reasons.append(
                        "missing qmin/qmax/qscale/qzeropoint "
                        f"counts={counts}"
                    )
                elif len(set(counts)) != 1:
                    reasons.append(f"inconsistent qinfo counts={counts}")
                else:
                    if not all(
                        np.all(np.isfinite(values))
                        for values in (qmin, qmax, qscale, qzero)
                    ):
                        reasons.append("qinfo contains non-finite values")
                    if np.any(qmax <= qmin):
                        reasons.append("qmax must be greater than qmin")
                    if np.any(qscale <= 0.0):
                        reasons.append("qscale must be positive")
            row["valid"] = not reasons
            if reasons:
                row["reason"] = "; ".join(reasons)
        except Exception as error:
            row.update(
                {
                    "dtype": "ERROR",
                    "shape": None,
                    "qinfo_count": 0,
                    "valid": False,
                    "reason": f"tensor lookup failed: {error}",
                }
            )
            dtype_counts["ERROR"] = dtype_counts.get("ERROR", 0) + 1
        if not row["valid"]:
            try:
                node = json.loads(context._GetAttr(name))
                row["layer_type"] = node.get("layerType")
                row["inputs"] = node.get("input", [])
            except Exception:
                row["layer_type"] = None
                row["inputs"] = []
            invalid.append(dict(row))
        rows.append(row)
    return {
        "format": "mage-qwen-prefill-tensor-audit-v1",
        "fb": str(path),
        "tensor_count": len(rows),
        "dtype_counts": dtype_counts,
        "invalid_int8_qinfo_count": len(invalid),
        "invalid_int8_qinfo": invalid,
        "ok": not invalid,
        "tensors": rows,
    }


def _quantize_uint8(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    value = np.asarray(array, dtype=np.float32)
    low = float(np.min(value))
    high = float(np.max(value))
    if low == high:
        epsilon = max(abs(low), 1.0) * 1e-6
        low -= epsilon
        high += epsilon
    encoded = np.clip(
        np.rint((value - low) * (255.0 / (high - low))), 0, 255
    ).astype(np.uint8)
    return np.ascontiguousarray(encoded), low, high


def build_quantlite_int8_layer_graph(
    config: QwenPrefillConfig,
    layer_id: int,
    seq_len: int,
    weights: dict[str, np.ndarray],
    range_json: str | Path,
    *,
    top_k: int = 0,
    attention_mode: str = "arm-causal-hxs",
    activation_granularity: str = "scalar",
    per_channel_qk_max_requant_multiplier: float = 0.99,
    softmax_threads: int = 0,
    token_group_boundaries: Iterable[int] = (),
    token_hybrid_qkv_start_layer: int | None = None,
    debug_outputs: Iterable[str] | None = None,
    addon_path: str | Path | None = None,
    create_executor: bool = False,
    use_hardware: bool = False,
) -> tuple[Any, Any | None, list[str], list[str], dict[str, Any]]:
    """Build explicit Q/DQ islands and quantize weights with QuantizeLite.

    Projection weights are encoded by QuantizeLite in isolated source
    Quantize->Conv staging islands. The complete Qwen graph is then built with
    the same explicit source Q/DQ topology used by Vit.py; the current SDK
    cannot scan the mixed Qwen CustomOp graph as one Lite pass. Explicit float
    boundaries use the built-in DeQuantize operator with an FP16 destination.
    ``arm-causal-hxs`` keeps QK/AV on the UINT8 path and executes only causal
    Softmax in the ARM custom operator. ``legacy-fp16`` retains the previous
    FP16 mask/Add/Softmax path for A/B diagnosis.
    """
    if not 0 <= layer_id < config.num_hidden_layers:
        raise ValueError(f"layer_id outside [0,{config.num_hidden_layers})")
    ranges = json.loads(Path(range_json).read_text())
    if attention_mode not in {
        "arm-causal-hxs", "arm-causal-scalar", "legacy-fp16"
    }:
        raise ValueError(f"unsupported attention mode: {attention_mode}")
    if activation_granularity not in {"scalar", "token"}:
        raise ValueError(
            "activation_granularity must be 'scalar' or 'token'"
        )
    token_quantization = activation_granularity == "token"
    if token_quantization:
        raise ValueError(
            "token MatMul activation qinfo cannot be combined with the "
            "per-output-channel projection weights required by the current "
            "SDK. Use scalar; token-role hybridization is implemented with "
            "separate Conv1x1 source branches, as in Vit.py."
        )
    arm_causal = attention_mode.startswith("arm-causal-")
    arm_causal_hxs = attention_mode == "arm-causal-hxs"
    if softmax_threads < 0:
        raise ValueError("softmax_threads must be non-negative")
    boundaries = tuple(int(value) for value in token_group_boundaries)
    if boundaries != tuple(sorted(set(boundaries))) or any(
        value <= 0 or value >= seq_len for value in boundaries
    ):
        raise ValueError(
            "token_group_boundaries must be unique, sorted values inside "
            f"(0, {seq_len})"
        )
    token_group_specs: tuple[tuple[str, int, int], ...] = ()
    if boundaries:
        stops = (0, *boundaries, seq_len)
        token_group_specs = tuple(
            (f"g{index}", begin, end)
            for index, (begin, end) in enumerate(zip(stops, stops[1:]))
        )
    if token_hybrid_qkv_start_layer is not None and not (
        0 <= int(token_hybrid_qkv_start_layer) < config.num_hidden_layers
    ):
        raise ValueError(
            "token_hybrid_qkv_start_layer must name a decoder layer"
        )
    split_qkv = bool(
        token_group_specs
        and token_hybrid_qkv_start_layer is not None
        and layer_id >= int(token_hybrid_qkv_start_layer)
    )
    max_multiplier = float(per_channel_qk_max_requant_multiplier)
    if not (np.isfinite(max_multiplier) and 0.0 < max_multiplier < 1.0):
        raise ValueError(
            "per-channel QK max requant multiplier must be in (0, 1)"
        )
    selection = select_outlier_branches(ranges, top_k)
    fp_attention = layer_id in selection["fp_attn_layers"]
    fp_mlp = layer_id in selection["fp_mlp_layers"]
    Op, TFContext, TFExecutor, TFDataType = _load_tfdl(addon_path)
    prefix = f"layers.{layer_id}"
    ctx = TFContext(f"MageQwenPrefillInt8Layer{layer_id}Seq{seq_len}")

    qk_row_min: list[float] = []
    qk_row_max: list[float] = []
    score_row_min: list[float] = []
    score_row_max: list[float] = []
    score_global_min = 0.0
    score_global_max = 1.0
    qk_scalar_min = 0.0
    qk_scalar_max = 1.0
    expanded_qk_rows = 0
    maximum_multiplier_before = 0.0
    maximum_multiplier_after = 0.0
    if arm_causal_hxs:
        row_name = f"{prefix}.self_attn.qk_matmul.rows"
        item = ranges.get(row_name)
        if not isinstance(item, dict) or "min" not in item or "max" not in item:
            raise KeyError(
                "arm-causal-hxs requires true H*S QK ranges; missing "
                f"{row_name}. Regenerate calibration with the current "
                "evaluate_qwen_prefill.py."
            )
        qk_min_array = np.asarray(item["min"], dtype=np.float64)
        qk_max_array = np.asarray(item["max"], dtype=np.float64)
        expected_rows = config.num_attention_heads * seq_len
        if qk_min_array.shape != (expected_rows,) or qk_max_array.shape != (
            expected_rows,
        ):
            raise ValueError(
                f"{row_name} has {qk_min_array.size} rows, expected "
                f"H*S={config.num_attention_heads}*{seq_len}={expected_rows}"
            )
        q_min, q_max = _range_values(ranges, f"{prefix}.self_attn.q_rope")
        k_min, k_max = _range_values(ranges, f"{prefix}.self_attn.k_rope")
        accumulator_scale = (
            (q_max - q_min) * (k_max - k_min) / (255.0 * 255.0)
        )
        row_scales = (qk_max_array - qk_min_array) / 255.0
        if np.any(row_scales <= 0.0):
            raise ValueError(f"{row_name} contains a non-positive row scale")
        minimum_row_scale = float(
            np.nextafter(
                np.float32(accumulator_scale / max_multiplier),
                np.float32(np.inf),
            )
        )
        expand = row_scales < minimum_row_scale
        expanded_qk_rows = int(np.count_nonzero(expand))
        maximum_multiplier_before = float(
            np.max(accumulator_scale / row_scales)
        )
        if expanded_qk_rows:
            factors = np.ones_like(row_scales)
            factors[expand] = minimum_row_scale / row_scales[expand]
            qk_min_array *= factors
            qk_max_array *= factors
            row_scales = (qk_max_array - qk_min_array) / 255.0
        maximum_multiplier_after = float(
            np.max(accumulator_scale / row_scales)
        )
        attention_scale = config.head_dim**-0.5
        qk_row_min = qk_min_array.astype(np.float32).tolist()
        qk_row_max = qk_max_array.astype(np.float32).tolist()
        score_row_min = (
            qk_min_array * attention_scale
        ).astype(np.float32).tolist()
        score_row_max = (
            qk_max_array * attention_scale
        ).astype(np.float32).tolist()
        score_global_min = float(np.min(qk_min_array) * attention_scale)
        score_global_max = float(np.max(qk_max_array) * attention_scale)
        print(
            f"[QWEN-QK-SCALE-FLOOR] layer={layer_id:02d} "
            f"expanded={expanded_qk_rows}/{expected_rows} "
            f"max_multiplier={maximum_multiplier_before:.6g}->"
            f"{maximum_multiplier_after:.6g}"
        )
    elif arm_causal:
        qk_scalar_min, qk_scalar_max = _range_values(
            ranges, f"{prefix}.self_attn.qk_matmul"
        )
        attention_scale = config.head_dim**-0.5
        score_global_min = qk_scalar_min * attention_scale
        score_global_max = qk_scalar_max * attention_scale

    fp16_parameters = {
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
    }
    if fp_attention:
        fp16_parameters.add(f"{prefix}.self_attn.o_proj.weight")
    if fp_mlp:
        fp16_parameters.update(
            {
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    int8_parameter_names = sorted(set(weights) - fp16_parameters)

    # Qwen has three CustomOp classes in its source graph.  The current SDK's
    # QuantizeLite scans those nodes and segfaults even in legacy scalar
    # attention mode, while Vit.py's standard-op graph can use a whole-graph
    # Lite pass.  Keep Vit's explicit source Q/DQ topology in the final graph,
    # but encode projection weights in isolated source Quantize->Conv islands.
    from TFDL2 import CalibrationMode, TFCalibration

    quant_ctx = TFContext(
        f"MageQwenPrefillQuantLiteWeightsLayer{layer_id}"
    )
    staging_params: dict[str, np.ndarray] = {}
    for name in int8_parameter_names:
        value = weights[name]
        staging_params[_conv_alias(name)] = np.ascontiguousarray(
            value.reshape(value.shape[0], value.shape[1], 1, 1),
            dtype=np.float32,
        )
    quant_ctx.RegisterParamToContext(**staging_params)
    staging_inputs: list[str] = []
    staging_outputs: list[str] = []
    staging_ranges: list[tuple[str, str]] = []

    def projection_input_range(suffix: str) -> str:
        if suffix.startswith(
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        ):
            return f"{prefix}.input_norm"
        if suffix.startswith("self_attn.o_proj"):
            return f"{prefix}.self_attn.attention"
        if suffix.startswith(("mlp.gate_proj", "mlp.up_proj")):
            return f"{prefix}.post_norm"
        if suffix.startswith("mlp.down_proj"):
            return f"{prefix}.mlp.mid"
        raise KeyError(f"no QuantizeLite input range mapping for {suffix}")

    with quant_ctx:
        for name in int8_parameter_names:
            value = weights[name]
            suffix = name[len(prefix) + 1 : -len(".weight")]
            stage_input = Op.Placeholder2(
                quant_ctx,
                (1, int(value.shape[1]), 1, 1),
                TFDataType.TFDL_FLOAT,
            )
            stage_quantized = Op.Quantize(stage_input)
            stage_output = Op.Convolution2(
                stage_quantized,
                quant_ctx.GetParamSymbol(_conv_alias(name)),
                None,
                kernel=1,
                pad=0,
                stride=1,
                dilation=1,
                outChannel=int(value.shape[0]),
                group=1,
            )
            staging_inputs.append(str(stage_input))
            staging_outputs.append(str(stage_output))
            staging_ranges.extend(
                (
                    (str(stage_input), projection_input_range(suffix)),
                    (str(stage_quantized), projection_input_range(suffix)),
                    (str(stage_output), f"{prefix}.{suffix}"),
                )
            )
    quant_ctx.SetOutputs(staging_outputs)
    for symbol_name, range_name in staging_ranges:
        low, high = _range_values(ranges, range_name)
        if not quant_ctx.AddInt8Config(symbol_name, high, low):
            raise RuntimeError(
                f"QuantizeLite staging range failed: {symbol_name} <- "
                f"{range_name}"
            )
    weight_calibration = TFCalibration(
        quant_ctx,
        CalibrationMode.Naive,
        {"UseHardware": False, "FrugalMode": True},
    )
    weight_calibration.QuantizeLite(
        {name: TFDataType.TFDL_FLOAT for name in staging_inputs},
        MergeConcate=False,
        Perchannel=True,
    )
    quantized_parameters: dict[
        str, tuple[np.ndarray, list[float], list[float]]
    ] = {}
    for name in int8_parameter_names:
        parameter_name = _conv_alias(name)
        tensor = quant_ctx.GetParam(parameter_name)
        quantized_parameters[parameter_name] = (
            np.ascontiguousarray(tensor.toNumpy(), dtype=np.uint8),
            [float(value) for value in tensor.qmax],
            [float(value) for value in tensor.qmin],
        )

    registered: dict[str, np.ndarray] = {}
    for name, value in weights.items():
        parameter_name = _conv_alias(name) if value.ndim == 2 else name
        shaped = (
            value.reshape(value.shape[0], value.shape[1], 1, 1)
            if value.ndim == 2
            else value
        )
        if name in fp16_parameters:
            registered[parameter_name] = np.ascontiguousarray(
                shaped, dtype=np.float16
            )
    attention_scale_name = f"__qwen_prefill_quantlite_scale_{layer_id}"
    registered[attention_scale_name] = np.asarray(
        [config.head_dim**-0.5], dtype=np.float16
    )
    ctx.RegisterParamToContext(**registered)
    for parameter_name, (data, qmax, qmin) in quantized_parameters.items():
        if not ctx.RegisterQuantizedParamToContext(
            parameter_name, data, qmax, qmin
        ):
            raise RuntimeError(
                f"failed to register QuantizeLite parameter {parameter_name}"
            )

    symbols: dict[str, str] = {}
    symbol_range_sources: dict[str, str] = {}
    symbol_token_slices: dict[str, tuple[int, int]] = {}
    early_qinfo_symbols: set[str] = set()

    def scalar_activation_range(
        range_tag: str,
        token_slice: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        """Return the scalar range used by one explicit UINT8 island.

        Tok-hybrid branches use the union of the calibrated token rows that
        belong to that branch, matching the normal registration pass below.
        Keeping this calculation in one helper also lets UINT8 Swish/Mul get
        qinfo before TFContext.Close() validates their shapes.
        """
        source = (
            range_tag
            if range_tag.startswith("layers.")
            else f"{prefix}.{range_tag}"
        )
        low, high = _range_values(ranges, source)
        if token_slice is None:
            return low, high
        token_range = ranges.get(source + ".tokens")
        if not isinstance(token_range, dict):
            raise KeyError(
                f"Tok hybrid requires {source}.tokens in the range JSON"
            )
        begin, end = token_slice
        token_min = np.asarray(token_range["min"], dtype=np.float64)
        token_max = np.asarray(token_range["max"], dtype=np.float64)
        if token_min.shape != (seq_len,) or token_max.shape != (seq_len,):
            raise ValueError(
                f"{source}.tokens must contain S={seq_len} ranges"
            )
        return (
            float(np.min(token_min[begin:end])),
            float(np.max(token_max[begin:end])),
        )

    def register_qinfo_before_close(
        value: Any,
        range_tag: str,
        token_slice: tuple[int, int] | None = None,
    ) -> None:
        """Install qinfo needed by UINT8 elementwise shape inference.

        The rest of the graph keeps the existing post-construction range
        registration. UINT8 Swish and the Mul workaround are different:
        Close() itself requires input/output qinfo. Register their complete
        island eagerly and remember it so aliases are not added twice.
        """
        symbol_name = str(value)
        if symbol_name in early_qinfo_symbols:
            return
        low, high = scalar_activation_range(range_tag, token_slice)
        if not ctx.AddInt8Config(symbol_name, high, low):
            raise RuntimeError(
                f"failed to register pre-Close qinfo for {symbol_name} "
                f"from {range_tag}"
            )
        early_qinfo_symbols.add(symbol_name)

    def mark(
        tag: str,
        value: Any,
        range_tag: str | None = None,
        token_slice: tuple[int, int] | None = None,
    ) -> Any:
        logical = f"{prefix}.{tag}"
        symbols[logical] = str(value)
        if range_tag is not None:
            symbol_range_sources[logical] = (
                range_tag
                if range_tag.startswith("layers.")
                else f"{prefix}.{range_tag}"
            )
        if token_slice is not None:
            symbol_token_slices[logical] = token_slice
        return value

    def quantize(
        value: Any,
        tag: str,
        range_tag: str,
        token_slice: tuple[int, int] | None = None,
    ) -> Any:
        return mark(tag, Op.Quantize(value), range_tag, token_slice)

    def dequantize(value: Any, tag: str) -> Any:
        output = Op.DeQuantize(value, TFDataType.TFDL_FLOAT16)
        return mark(tag, output)

    def manual_projection(
        value: Any,
        suffix: str,
        in_channels: int,
        out_channels: int,
        range_tag: str,
        *,
        token_count: int = seq_len,
        projection_group: str | None = None,
        token_slice: tuple[int, int] | None = None,
        input_is_conv: bool = False,
        output_is_conv: bool = False,
    ) -> Any:
        weight_name = f"{prefix}.{suffix}.weight"
        projection_tag = (
            f"{suffix}.{projection_group}" if projection_group else suffix
        )
        if token_quantization:
            if input_is_conv or output_is_conv:
                raise ValueError(
                    "token MatMul projections require token-major tensors"
                )
            output = mark(
                projection_tag + ".matmul",
                Op.MatMul(
                    value,
                    ctx.GetParamSymbol(_matmul_alias(weight_name)),
                    transA=False,
                    transB=True,
                ),
                suffix,
                token_slice,
            )
            return mark(projection_tag, output, suffix, token_slice)
        if input_is_conv:
            conv_input = value
        else:
            grid_height, grid_width = _factor_token_grid(token_count)
            transposed = mark(
                projection_tag + ".transposed",
                Op.Transpose(value, (0, 2, 1)),
                range_tag,
                token_slice,
            )
            conv_input = mark(
                projection_tag + ".conv_input",
                Op.Reshape(
                    transposed,
                    (1, in_channels, grid_height, grid_width),
                ),
                range_tag,
                token_slice,
            )
        output = mark(
            projection_tag + ".conv",
            Op.Convolution2(
                conv_input,
                ctx.GetParamSymbol(_conv_alias(weight_name)),
                None,
                kernel=1,
                pad=0,
                stride=1,
                dilation=1,
                outChannel=out_channels,
                group=1,
            ),
            suffix,
            token_slice,
        )
        if output_is_conv:
            return mark(
                projection_tag, output, suffix, token_slice
            )
        token_view = mark(
            projection_tag + ".tokens_ncs",
            Op.Reshape(output, (1, out_channels, token_count)),
            suffix,
            token_slice,
        )
        return mark(
            projection_tag,
            Op.Transpose(token_view, (0, 2, 1)),
            suffix,
            token_slice,
        )

    def tokens_to_conv(
        value: Any,
        tag: str,
        channels: int,
        token_count: int,
        range_tag: str,
        token_slice: tuple[int, int] | None = None,
    ) -> Any:
        grid_height, grid_width = _factor_token_grid(token_count)
        transposed = mark(
            tag + ".transposed",
            Op.Transpose(value, (0, 2, 1)),
            range_tag,
            token_slice,
        )
        return mark(
            tag + ".conv_input",
            Op.Reshape(
                transposed,
                (1, channels, grid_height, grid_width),
            ),
            range_tag,
            token_slice,
        )

    def conv_to_tokens(
        value: Any,
        tag: str,
        channels: int,
        token_count: int,
        range_tag: str,
        token_slice: tuple[int, int] | None = None,
    ) -> Any:
        token_view = mark(
            tag + ".tokens_ncs",
            Op.Reshape(value, (1, channels, token_count)),
            range_tag,
            token_slice,
        )
        return mark(
            tag,
            Op.Transpose(token_view, (0, 2, 1)),
            range_tag,
            token_slice,
        )

    def conv_to_heads(
        value: Any,
        tag: str,
        heads: int,
        token_count: int,
        range_tag: str,
    ) -> Any:
        # NCHW Conv output flattens as [head, head_dim, token].  One reshape
        # plus one transpose reaches the native RoPE layout [B,H,S,D].
        reshaped = mark(
            tag + ".reshaped",
            Op.Reshape(
                value,
                (1, heads, config.head_dim, token_count),
            ),
            range_tag,
        )
        return mark(
            tag,
            Op.Transpose(reshaped, (0, 1, 3, 2)),
            range_tag,
        )

    with ctx:
        hidden = Op.Placeholder2(
            ctx,
            (1, seq_len, config.hidden_size),
            TFDataType.TFDL_FLOAT16,
        )
        rope_sin = Op.Placeholder2(
            ctx, (1, 1, seq_len, config.head_dim), TFDataType.TFDL_FLOAT
        )
        rope_cos = Op.Placeholder2(
            ctx, (1, 1, seq_len, config.head_dim), TFDataType.TFDL_FLOAT
        )
        symbols[f"{prefix}.input"] = str(hidden)
        symbols["rope_sin"] = str(rope_sin)
        symbols["rope_cos"] = str(rope_cos)
        input_names = [str(hidden), str(rope_sin), str(rope_cos)]
        mask = None
        if not arm_causal:
            mask = Op.Placeholder2(
                ctx, (1, seq_len, seq_len), TFDataType.TFDL_FLOAT16
            )
            symbols["causal_mask"] = str(mask)
            input_names.append(str(mask))

        input_norm, input_rms = _rms_norm(
            Op,
            ctx,
            hidden,
            f"{prefix}.input_layernorm.weight",
            config.rms_norm_eps,
            f"qwen_prefill_int8_l{layer_id}_input_rms",
        )
        mark("input_rms", input_rms)
        input_norm = mark("input_norm", input_norm)
        if split_qkv:
            # The three roles share one set of per-output-channel weights but
            # enter separate Q/DQ islands. This is the Qwen analogue of
            # Vit.py's CLS/register/patch Tok hybrid. Transpose the complete
            # FP16 sequence once, then slice its token axis in channel-major
            # layout. The former token-major implementation transposed every
            # group before all three projections and transposed every Q/K/V
            # result back afterwards, adding seven redundant Transpose nodes
            # for the two-group [prefix, rest] profile. Keeping the Slice on
            # the FP16 side remains intentional: UINT8 Slice fan-out crashes
            # this SDK.
            input_norm_ncs = mark(
                "input_norm.transposed",
                Op.Transpose(input_norm, (0, 2, 1)),
                "input_norm",
            )
            norm_groups = Op.Slice(
                input_norm_ncs,
                axis=2,
                split=tuple(
                    end - begin for _, begin, end in token_group_specs
                ),
            )
            projected_groups: dict[str, list[Any]] = {
                "q": [], "k": [], "v": []
            }
            projection_specs = (
                (
                    "q",
                    "self_attn.q_proj",
                    config.num_attention_heads,
                ),
                (
                    "k",
                    "self_attn.k_proj",
                    config.num_key_value_heads,
                ),
                (
                    "v",
                    "self_attn.v_proj",
                    config.num_key_value_heads,
                ),
            )
            for (group_name, begin, end), norm_group in zip(
                token_group_specs, norm_groups
            ):
                token_slice = (begin, end)
                token_count = end - begin
                grid_height, grid_width = _factor_token_grid(token_count)
                norm_conv = mark(
                    f"input_norm.{group_name}.conv_input",
                    Op.Reshape(
                        norm_group,
                        (
                            1,
                            config.hidden_size,
                            grid_height,
                            grid_width,
                        ),
                    ),
                    "input_norm",
                    token_slice,
                )
                quant_input = quantize(
                    norm_conv,
                    f"input_norm.quantized.{group_name}",
                    "input_norm",
                    token_slice,
                )
                for short_name, projection, heads in projection_specs:
                    out_channels = heads * config.head_dim
                    projected = manual_projection(
                        quant_input,
                        projection,
                        config.hidden_size,
                        out_channels,
                        "input_norm",
                        token_count=end - begin,
                        projection_group=group_name,
                        token_slice=token_slice,
                        input_is_conv=True,
                        output_is_conv=True,
                    )
                    projected_fp16 = dequantize(
                        projected, f"{projection}.{group_name}.fp16"
                    )
                    projected_groups[short_name].append(
                        mark(
                            f"{projection}.{group_name}.heads_nhds",
                            Op.Reshape(
                                projected_fp16,
                                (
                                    1,
                                    heads,
                                    config.head_dim,
                                    token_count,
                                ),
                            ),
                        )
                    )
            q_nhds = mark(
                "self_attn.q_proj.fp16",
                Op.Concat(tuple(projected_groups["q"]), axis=3),
            )
            k_nhds = mark(
                "self_attn.k_proj.fp16",
                Op.Concat(tuple(projected_groups["k"]), axis=3),
            )
            v_nhds = mark(
                "self_attn.v_proj.fp16",
                Op.Concat(tuple(projected_groups["v"]), axis=3),
            )
            # Q/K need [B,H,S,D] for RMSNorm and RoPE; V needs the same
            # logical layout for AV and cache export. These three final
            # layout conversions are necessary and now operate on the full
            # S=1024 tensor, whose dimensions satisfy NPU Transpose tiling.
            q = Op.Transpose(q_nhds, (0, 1, 3, 2))
            k = Op.Transpose(k_nhds, (0, 1, 3, 2))
            v = Op.Transpose(v_nhds, (0, 1, 3, 2))
        else:
            input_norm_conv = tokens_to_conv(
                input_norm,
                "input_norm",
                config.hidden_size,
                seq_len,
                "input_norm",
            )
            quant_input = quantize(
                input_norm_conv,
                "input_norm.quantized",
                "input_norm",
            )
            q = manual_projection(
                quant_input,
                "self_attn.q_proj",
                config.hidden_size,
                config.query_size,
                "input_norm",
                input_is_conv=True,
                output_is_conv=True,
            )
            k = manual_projection(
                quant_input,
                "self_attn.k_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                "input_norm",
                input_is_conv=True,
                output_is_conv=True,
            )
            v = manual_projection(
                quant_input,
                "self_attn.v_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                "input_norm",
                input_is_conv=True,
                output_is_conv=True,
            )
            q = dequantize(q, "self_attn.q_proj.fp16")
            k = dequantize(k, "self_attn.k_proj.fp16")
            # Q/K must enter FP16 RMSNorm and RoPE. V has no floating
            # operation before AV, so keep its projection in UINT8 for the
            # ARM attention path and dequantize only the exported cache.
            if not arm_causal:
                v = dequantize(v, "self_attn.v_proj.fp16")
            q = conv_to_heads(
                q,
                "self_attn.q_proj.heads",
                config.num_attention_heads,
                seq_len,
                "self_attn.q_proj",
            )
            k = conv_to_heads(
                k,
                "self_attn.k_proj.heads",
                config.num_key_value_heads,
                seq_len,
                "self_attn.k_proj",
            )
            v = conv_to_heads(
                v,
                "self_attn.v_proj.heads",
                config.num_key_value_heads,
                seq_len,
                "self_attn.v_proj",
            )
        q, q_rms = _rms_norm(
            Op,
            ctx,
            q,
            f"{prefix}.self_attn.q_norm.weight",
            config.rms_norm_eps,
            f"qwen_prefill_int8_l{layer_id}_q_rms",
        )
        k, k_rms = _rms_norm(
            Op,
            ctx,
            k,
            f"{prefix}.self_attn.k_norm.weight",
            config.rms_norm_eps,
            f"qwen_prefill_int8_l{layer_id}_k_rms",
        )
        mark("self_attn.q_rms", q_rms)
        mark("self_attn.k_rms", k_rms)
        q = mark("self_attn.q_norm", q)
        k = mark("self_attn.k_norm", k)
        q = mark("self_attn.q_heads", q)
        k = mark("self_attn.k_heads", k)
        v = mark(
            "self_attn.v_heads", v,
            "self_attn.v_cache" if arm_causal and not split_qkv else None,
        )
        rope = Op.Custom(
            (q, k, rope_sin, rope_cos),
            (
                f"qwen_prefill_int8_l{layer_id}_q_rope",
                f"qwen_prefill_int8_l{layer_id}_k_rope",
            ),
            "ApplyRope",
            "{}",
        )
        q = mark("self_attn.q_rope", rope[0], "self_attn.q_rope")
        k = mark("self_attn.k_rope", rope[1], "self_attn.k_rope")
        v = mark("self_attn.v_cache", v, "self_attn.v_cache")

        # K already passed through FP16 RoPE. V may still be the original
        # UINT8 projection; its cache DQ is deliberately deferred until the
        # terminal output branch so AV does not see a DQ->Q round trip.
        k_cache_fp16 = k
        v_cache_export_source = v
        v_cache_export_needs_dequant = arm_causal and not split_qkv
        if arm_causal:
            q = quantize(
                q, "self_attn.q_rope.quantized", "self_attn.q_rope"
            )
            k = quantize(
                k, "self_attn.k_rope.quantized", "self_attn.k_rope"
            )
            if split_qkv:
                # Tok-hybrid V groups have distinct scalar qinfo. Their FP16
                # Concat still needs one common AV qinfo; later work can
                # replace this boundary with per-group Requant LUTs.
                v = quantize(
                    v, "self_attn.v_cache.quantized", "self_attn.v_cache"
                )

        if arm_causal:
            # Do not materialize repeated GQA K/V heads.  The SDK's UINT8
            # Slice -> repeated Concat path crashes when its input is a
            # Quantize output.  Flatten each KV head's four query heads into
            # the M dimension instead: [8,4*S,D] @ [8,D,S].  Flattened row
            # order remains exactly [query_head, query_token], so the H*S
            # calibration order is unchanged and K/V storage is 4x smaller.
            repeated_k = mark(
                "self_attn.k_repeated", k, "self_attn.k_rope"
            )
            repeated_v = mark(
                "self_attn.v_repeated", v, "self_attn.v_cache"
            )
            grouped_rows = config.kv_repeats * seq_len
            q3 = mark(
                "self_attn.q_matmul_input",
                Op.Reshape(
                    q,
                    (
                        config.num_key_value_heads,
                        grouped_rows,
                        config.head_dim,
                    ),
                ),
                "self_attn.q_rope",
            )
            k3_reshaped = mark(
                "self_attn.k_matmul_input.reshaped",
                Op.Reshape(
                    k,
                    (
                        config.num_key_value_heads,
                        seq_len,
                        config.head_dim,
                    ),
                ),
                "self_attn.k_rope",
            )
            k3 = mark(
                "self_attn.k_matmul_input",
                Op.Transpose(k3_reshaped, (0, 2, 1)),
                "self_attn.k_rope",
            )
            v3 = mark(
                "self_attn.v_matmul_input",
                Op.Reshape(
                    v,
                    (
                        config.num_key_value_heads,
                        seq_len,
                        config.head_dim,
                    ),
                ),
                "self_attn.v_cache",
            )
            qk_grouped = mark(
                "self_attn.qk_matmul.grouped",
                Op.MatMul(q3, k3, transA=False, transB=False),
                "self_attn.qk_matmul",
            )
            qk = mark(
                "self_attn.qk_matmul",
                qk_grouped,
                "self_attn.qk_matmul",
            )
        else:
            repeated_k = mark(
                "self_attn.k_repeated",
                _repeat_gqa(Op, k, config),
                "self_attn.k_rope",
            )
            repeated_v = mark(
                "self_attn.v_repeated",
                _repeat_gqa(Op, v, config),
                "self_attn.v_cache",
            )
            q3 = mark(
                "self_attn.q_matmul_input",
                Op.Reshape(
                    q,
                    (config.num_attention_heads, seq_len, config.head_dim),
                ),
                "self_attn.q_rope",
            )
            k3 = mark(
                "self_attn.k_matmul_input",
                Op.Transpose(
                    Op.Reshape(
                        repeated_k,
                        (
                            config.num_attention_heads,
                            seq_len,
                            config.head_dim,
                        ),
                    ),
                    (0, 2, 1),
                ),
                "self_attn.k_rope",
            )
            v3 = mark(
                "self_attn.v_matmul_input",
                Op.Reshape(
                    repeated_v,
                    (
                        config.num_attention_heads,
                        seq_len,
                        config.head_dim,
                    ),
                ),
                "self_attn.v_cache",
            )
            qk = mark(
                "self_attn.qk_matmul",
                Op.MatMul(q3, k3, transA=False, transB=False),
                "self_attn.qk_matmul",
            )
        if arm_causal:
            # Requant keeps every raw code unchanged while folding 1/sqrt(D)
            # into its qinfo.  The CustomOp consumes scalar or H*S qinfo
            # directly; no transport conversion or scale sidecar is needed.
            scores = mark(
                "self_attn.scores",
                Op.Requantize(qk, list(range(256))),
                "self_attn.scores",
            )
            probability_value = Op.Custom(
                (scores,),
                (f"qwen_prefill_l{layer_id}_arm_causal_probabilities",),
                "ArmCausalMaskSoftmax",
                json.dumps({"threads": int(softmax_threads)}),
            )
            if isinstance(probability_value, (tuple, list)):
                probability_value = probability_value[0]
            probabilities = mark(
                "self_attn.probabilities",
                probability_value,
                "self_attn.probabilities",
            )
            symbols[f"{prefix}.self_attn.masked_scores"] = str(scores)
        else:
            scores = mark(
                "self_attn.scores",
                Op.Mul(qk, ctx.GetParamSymbol(attention_scale_name)),
                "self_attn.scores",
            )
            assert mask is not None
            masked = mark(
                "self_attn.masked_scores", Op.Add(scores, mask)
            )
            probabilities = mark(
                "self_attn.probabilities", Op.Softmax(masked, axis=2)
            )
        if arm_causal:
            probability_grouped = mark(
                "self_attn.probabilities.grouped",
                probabilities,
                "self_attn.probabilities",
            )
            attention_grouped = mark(
                "self_attn.attention.grouped",
                Op.MatMul(
                    probability_grouped, v3,
                    transA=False, transB=False,
                ),
                "self_attn.attention",
            )
            attention = mark(
                "self_attn.attention",
                Op.Reshape(
                    attention_grouped,
                    (
                        config.num_attention_heads,
                        seq_len,
                        config.head_dim,
                    ),
                ),
                "self_attn.attention",
            )
        else:
            attention = mark(
                "self_attn.attention",
                Op.MatMul(
                    probabilities, v3, transA=False, transB=False
                ),
                "self_attn.attention",
            )
        if fp_attention and arm_causal:
            attention = dequantize(
                attention, "self_attn.attention_tokens.fp16"
            )
        grid_height, grid_width = _factor_token_grid(seq_len)
        attention_transposed = mark(
            "self_attn.attention_tokens.transposed",
            Op.Transpose(attention, (0, 2, 1)),
            "self_attn.attention",
        )
        attention = mark(
            "self_attn.attention_tokens",
            Op.Reshape(
                attention_transposed,
                (1, config.query_size, grid_height, grid_width),
            ),
            "self_attn.attention",
        )
        if not fp_attention and not arm_causal:
            attention = quantize(
                attention,
                "self_attn.attention_tokens.quantized",
                "self_attn.attention",
            )
        # AV is UINT8.  Splitting it here triggers the SDK's known UINT8
        # Slice->multi-branch crash, so o_proj intentionally retains the
        # single scalar-qinfo path; Tok hybrid starts only from FP16 tensors.
        projected = manual_projection(
            attention,
            "self_attn.o_proj",
            config.query_size,
            config.hidden_size,
            "self_attn.attention",
            input_is_conv=True,
        )
        if not fp_attention:
            projected = dequantize(projected, "self_attn.o_proj.fp16")
        residual = mark("attn_residual", Op.Add(hidden, projected))

        post_norm, post_rms = _rms_norm(
            Op,
            ctx,
            residual,
            f"{prefix}.post_attention_layernorm.weight",
            config.rms_norm_eps,
            f"qwen_prefill_int8_l{layer_id}_post_rms",
        )
        mark("post_rms", post_rms)
        post_norm = mark("post_norm", post_norm)
        if token_group_specs and not fp_mlp:
            # Vit.py's Tok hybrid runs semantic token roles through separate
            # MLP Conv branches.  Each Q/DQ island below receives the min/max
            # union for only its contiguous prompt group, while all groups
            # share the same per-output-channel quantized weights.
            post_groups = Op.Slice(
                post_norm,
                axis=1,
                split=tuple(
                    end - begin for _, begin, end in token_group_specs
                ),
            )
            down_groups = []
            for (group_name, begin, end), post_group in zip(
                token_group_specs, post_groups
            ):
                token_slice = (begin, end)
                token_count = end - begin
                post_conv = tokens_to_conv(
                    post_group,
                    f"post_norm.{group_name}",
                    config.hidden_size,
                    token_count,
                    "post_norm",
                    token_slice,
                )
                mlp_input = quantize(
                    post_conv,
                    f"post_norm.quantized.{group_name}",
                    "post_norm",
                    token_slice,
                )
                gate = manual_projection(
                    mlp_input,
                    "mlp.gate_proj",
                    config.hidden_size,
                    config.intermediate_size,
                    "post_norm",
                    token_count=token_count,
                    projection_group=group_name,
                    token_slice=token_slice,
                    input_is_conv=True,
                    output_is_conv=True,
                )
                up = manual_projection(
                    mlp_input,
                    "mlp.up_proj",
                    config.hidden_size,
                    config.intermediate_size,
                    "post_norm",
                    token_count=token_count,
                    projection_group=group_name,
                    token_slice=token_slice,
                    input_is_conv=True,
                    output_is_conv=True,
                )
                # Keep the entire SwiGLU/down-projection compute island in
                # UINT8. Both Swish and the equal-shape tensor/tensor Mul now
                # use native SDK operators and remain eligible for NPU
                # execution.
                register_qinfo_before_close(
                    gate, "mlp.gate_proj", token_slice
                )
                register_qinfo_before_close(
                    up, "mlp.up_proj", token_slice
                )
                gate = mark(
                    f"mlp.gate_silu.{group_name}",
                    Op.Swish(gate),
                    "mlp.gate_silu",
                    token_slice,
                )
                register_qinfo_before_close(
                    gate, "mlp.gate_silu", token_slice
                )
                mul_value = Op.Mul(gate, up)
                mlp_mid = mark(
                    f"mlp.quantized_mul.{group_name}",
                    mul_value,
                    "mlp.mid",
                    token_slice,
                )
                mlp_mid = mark(
                    f"mlp.mid.{group_name}",
                    mlp_mid,
                    "mlp.mid",
                    token_slice,
                )
                register_qinfo_before_close(
                    mlp_mid, "mlp.mid", token_slice
                )
                down_group = manual_projection(
                    mlp_mid,
                    "mlp.down_proj",
                    config.intermediate_size,
                    config.hidden_size,
                    "mlp.mid",
                    token_count=token_count,
                    projection_group=group_name,
                    token_slice=token_slice,
                    input_is_conv=True,
                )
                down_groups.append(
                    dequantize(
                        down_group,
                        f"mlp.down_proj.{group_name}.fp16",
                    )
                )
            down = mark("mlp.down_proj", Op.Concat(tuple(down_groups), axis=1))
        else:
            mlp_input = tokens_to_conv(
                post_norm,
                "post_norm",
                config.hidden_size,
                seq_len,
                "post_norm",
            )
            if not fp_mlp:
                mlp_input = quantize(
                    mlp_input, "post_norm.quantized", "post_norm"
                )
            gate = manual_projection(
                mlp_input,
                "mlp.gate_proj",
                config.hidden_size,
                config.intermediate_size,
                "post_norm",
                input_is_conv=True,
                output_is_conv=True,
            )
            up = manual_projection(
                mlp_input,
                "mlp.up_proj",
                config.hidden_size,
                config.intermediate_size,
                "post_norm",
                input_is_conv=True,
                output_is_conv=True,
            )
            if not fp_mlp:
                register_qinfo_before_close(gate, "mlp.gate_proj")
                register_qinfo_before_close(up, "mlp.up_proj")
            if fp_mlp:
                gate = mark(
                    "mlp.gate_silu", Op.Swish(gate), "mlp.gate_silu"
                )
                mlp_mid = mark(
                    "mlp.mid", Op.Mul(gate, up), "mlp.mid"
                )
            else:
                gate = mark(
                    "mlp.gate_silu", Op.Swish(gate), "mlp.gate_silu"
                )
                register_qinfo_before_close(gate, "mlp.gate_silu")
                mul_value = Op.Mul(gate, up)
                mlp_mid = mark(
                    "mlp.quantized_mul", mul_value, "mlp.mid"
                )
                mlp_mid = mark("mlp.mid", mlp_mid, "mlp.mid")
                register_qinfo_before_close(mlp_mid, "mlp.mid")
            down = manual_projection(
                mlp_mid,
                "mlp.down_proj",
                config.intermediate_size,
                config.hidden_size,
                "mlp.mid",
                input_is_conv=True,
            )
            if not fp_mlp:
                down = dequantize(down, "mlp.down_proj.fp16")
        output = mark("output", Op.Add(residual, down))
        v_cache_fp16 = (
            dequantize(
                v_cache_export_source, "self_attn.v_cache.fp16"
            )
            if v_cache_export_needs_dequant
            else v_cache_export_source
        )
        cache_shape = (
            1,
            config.num_key_value_heads,
            seq_len,
            config.head_dim,
        )
        # Reshape provides a terminal alias without arithmetic or additional
        # storage. K still needs a terminal consumer because it originates
        # from ApplyRope, whose direct output is not stable across FB reload.
        k_fp16 = mark(
            "self_attn.k_export", Op.Reshape(k_cache_fp16, cache_shape)
        )
        v_fp16 = mark(
            "self_attn.v_export", Op.Reshape(v_cache_fp16, cache_shape)
        )
        output_names = [str(output), str(k_fp16), str(v_fp16)]

    # QuantizeLite must only see q-info on tensors that are explicitly part of
    # an INT8 projection island.  Registering q-info on FP16 attention nodes
    # makes the Lite pass try to quantize weightless operators as well.
    int8_suffixes = {
        "input_norm.quantized",
        "post_norm.quantized",
        "self_attn.attention_tokens.quantized",
    }
    if arm_causal:
        int8_suffixes.update(
            {
                "self_attn.q_rope.quantized",
                "self_attn.k_rope.quantized",
                "self_attn.v_cache.quantized",
                "self_attn.k_repeated",
                "self_attn.v_repeated",
                "self_attn.q_matmul_input",
                "self_attn.k_matmul_input.reshaped",
                "self_attn.k_matmul_input",
                "self_attn.v_matmul_input",
                "self_attn.qk_matmul.grouped",
                "self_attn.qk_matmul",
                "self_attn.scores",
                "self_attn.probabilities",
                "self_attn.probabilities.grouped",
                "self_attn.attention.grouped",
                "self_attn.attention",
                "self_attn.attention_tokens.transposed",
                "self_attn.attention_tokens",
            }
        )
        if not split_qkv:
            int8_suffixes.update(
                {
                    "self_attn.v_proj.heads",
                    "self_attn.v_proj.heads.reshaped",
                    "self_attn.v_heads",
                    "self_attn.v_cache",
                }
            )
    int8_projections = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ]
    if not fp_attention:
        int8_projections.append("self_attn.o_proj")
    if not fp_mlp:
        int8_projections.extend(
            ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        )
        int8_suffixes.update(
            {"mlp.gate_silu", "mlp.quantized_mul", "mlp.mid"}
        )
    for projection in int8_projections:
        int8_suffixes.update(
            {
                projection + ".transposed",
                projection + ".conv_input",
                projection + ".conv",
                projection + ".tokens_ncs",
                projection + ".matmul",
                projection,
            }
        )
    for group_name, _, _ in token_group_specs:
        int8_suffixes.update(
            {
                f"post_norm.quantized.{group_name}",
                f"mlp.gate_silu.{group_name}",
                f"mlp.quantized_mul.{group_name}",
                f"mlp.mid.{group_name}",
            }
        )
        for projection in (
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            group_projection = f"{projection}.{group_name}"
            int8_suffixes.update(
                {
                    group_projection,
                    group_projection + ".transposed",
                    group_projection + ".conv_input",
                    group_projection + ".conv",
                    group_projection + ".tokens_ncs",
                    group_projection + ".matmul",
                }
            )
        if split_qkv:
            int8_suffixes.add(f"input_norm.quantized.{group_name}")
            for projection in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ):
                group_projection = f"{projection}.{group_name}"
                int8_suffixes.update(
                    {
                        group_projection,
                        group_projection + ".transposed",
                        group_projection + ".conv_input",
                        group_projection + ".conv",
                        group_projection + ".tokens_ncs",
                        group_projection + ".matmul",
                    }
                )

    # Register activation ranges after all symbols have stable names.
    special_hxs_symbols = {
        f"{prefix}.self_attn.qk_matmul.grouped",
        f"{prefix}.self_attn.qk_matmul",
        f"{prefix}.self_attn.scores",
        f"{prefix}.self_attn.probabilities",
    }
    token_input_suffixes = {
        "input_norm.quantized",
        "post_norm.quantized",
        "self_attn.attention_tokens.quantized",
    }
    post_close_qinfo_symbols: set[str] = set()
    for logical, source in symbol_range_sources.items():
        suffix = logical[len(prefix) + 1 :]
        if suffix not in int8_suffixes:
            continue
        if arm_causal and logical in special_hxs_symbols:
            continue
        if source not in ranges:
            continue
        low, high = _range_values(ranges, source)
        token_range = ranges.get(source + ".tokens")
        token_slice = symbol_token_slices.get(logical)
        if token_slice is not None:
            if not isinstance(token_range, dict):
                raise KeyError(
                    f"Tok hybrid requires {source}.tokens in the range JSON"
                )
            begin, end = token_slice
            token_min = np.asarray(token_range["min"], dtype=np.float64)
            token_max = np.asarray(token_range["max"], dtype=np.float64)
            if token_min.shape != (seq_len,) or token_max.shape != (seq_len,):
                raise ValueError(
                    f"{source}.tokens must contain S={seq_len} ranges"
                )
            low = float(np.min(token_min[begin:end]))
            high = float(np.max(token_max[begin:end]))
        symbol_name = symbols[logical]
        if symbol_name in post_close_qinfo_symbols:
            continue
        if (
            token_quantization
            and suffix in token_input_suffixes
            and isinstance(token_range, dict)
        ):
            token_min = [float(value) for value in token_range["min"]]
            token_max = [float(value) for value in token_range["max"]]
            if len(token_min) != seq_len or len(token_max) != seq_len:
                raise ValueError(
                    f"{source}.tokens has {len(token_min)} rows, expected "
                    f"S={seq_len}"
                )
            registered_range = ctx.AddInt8ConfigPerChannel(
                symbol_name, token_max, token_min
            )
        else:
            registered_range = ctx.AddInt8Config(
                symbol_name, high, low
            )
        if not registered_range:
            raise RuntimeError(f"failed to register {logical} from {source}")
        post_close_qinfo_symbols.add(symbol_name)
    debug_suffixes = {
        (name.split(f"{prefix}.", 1)[1] if name.startswith(f"{prefix}.") else name)
        for name in (debug_outputs or ())
    }
    attention_or_later_prefixes = (
        "self_attn.qk_matmul",
        "self_attn.scores",
        "self_attn.masked_scores",
        "self_attn.probabilities",
        "self_attn.attention",
        "self_attn.o_proj",
        "attn_residual",
        "post_",
        "mlp.",
        "output",
    )
    need_hxs_qinfo = not debug_suffixes or any(
        suffix.startswith(attention_or_later_prefixes)
        for suffix in debug_suffixes
    )
    if arm_causal and need_hxs_qinfo:
        qk_symbols = [symbols[f"{prefix}.self_attn.qk_matmul"]]
        grouped_qk = symbols.get(
            f"{prefix}.self_attn.qk_matmul.grouped"
        )
        if grouped_qk is not None:
            qk_symbols.append(grouped_qk)
        for qk_symbol in dict.fromkeys(qk_symbols):
            if arm_causal_hxs:
                qk_registered = ctx.AddInt8ConfigPerChannel(
                    qk_symbol, qk_row_max, qk_row_min
                )
            else:
                qk_registered = ctx.AddInt8Config(
                    qk_symbol, qk_scalar_max, qk_scalar_min
                )
            if not qk_registered:
                raise RuntimeError("failed to register QK qinfo")
        score_symbol = symbols[f"{prefix}.self_attn.scores"]
        if arm_causal_hxs:
            score_registered = ctx.AddInt8ConfigPerChannel(
                score_symbol, score_row_max, score_row_min
            )
        else:
            score_registered = ctx.AddInt8Config(
                score_symbol, score_global_max, score_global_min
            )
        if not score_registered:
            raise RuntimeError("failed to register score boundary qinfo")
        if not ctx.AddInt8Config(
            symbols[f"{prefix}.self_attn.probabilities"], 1.0, 0.0
        ):
            raise RuntimeError("failed to register probability qinfo")
    selected_output_names = output_names
    if debug_outputs:
        selected_output_names = []
        for name in debug_outputs:
            logical = (
                name if name.startswith("layers.") else f"{prefix}.{name}"
            )
            if logical not in symbols:
                raise KeyError(f"unknown debug output {logical}")
            selected_output_names.append(symbols[logical])
    # Close the production graph with its stable hidden/K/V ABI.  A diagnostic
    # output is selected before QuantizeLite so unreachable suffixes are
    # actually absent from the pass; this also makes node-class bisection of a
    # failing SDK build deterministic.
    quantize_output_names = (
        selected_output_names if debug_outputs else output_names
    )
    ctx.SetOutputs(quantize_output_names)

    # Vit.py treats these tensors as the explicit floating side of an INT8
    # island.  QuantizeLite does not rewrite the graph, but giving it the same
    # audit fence prevents a future SDK optimization from absorbing RMSNorm,
    # RoPE, residuals, cache exports, or a selected Top-K branch.
    source_float_suffixes = {
        "input",
        "input_rms",
        "input_norm",
        "input_norm.transposed",
        "input_norm.conv_input",
        "self_attn.q_proj.fp16",
        "self_attn.k_proj.fp16",
        "self_attn.q_rms",
        "self_attn.k_rms",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "self_attn.q_heads",
        "self_attn.k_heads",
        "self_attn.q_rope",
        "self_attn.k_rope",
        "self_attn.v_cache.fp16",
        "self_attn.attention_tokens.fp16",
        "self_attn.o_proj.fp16",
        "attn_residual",
        "post_rms",
        "post_norm",
        "post_norm.transposed",
        "post_norm.conv_input",
        "mlp.down_proj.fp16",
        "output",
        "self_attn.k_export",
        "self_attn.v_export",
    }
    if split_qkv or not arm_causal:
        source_float_suffixes.update(
            {
                "self_attn.v_proj.fp16",
                "self_attn.v_heads",
                "self_attn.v_cache",
            }
        )
    for group_name, _, _ in token_group_specs:
        source_float_suffixes.update(
            {
                f"post_norm.{group_name}.transposed",
                f"post_norm.{group_name}.conv_input",
            }
        )
        if split_qkv:
            source_float_suffixes.update(
                {
                    f"input_norm.{group_name}.transposed",
                    f"input_norm.{group_name}.conv_input",
                }
            )
    if fp_attention:
        source_float_suffixes.update(
            {
                "self_attn.o_proj.transposed",
                "self_attn.o_proj.conv_input",
                "self_attn.o_proj.conv",
                "self_attn.o_proj",
            }
        )
    if fp_mlp:
        source_float_suffixes.update({"mlp.gate_silu", "mlp.mid"})
        for projection in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            source_float_suffixes.update(
                {
                    projection + ".transposed",
                    projection + ".conv_input",
                    projection + ".conv",
                    projection,
                }
            )
    avoid_tensors = tuple(
        dict.fromkeys(
            symbols[f"{prefix}.{suffix}"]
            for suffix in source_float_suffixes
            if f"{prefix}.{suffix}" in symbols
        )
    )
    ctx.source_float_tensors = avoid_tensors

    ctx.SetOutputs(quantize_output_names)

    parameter_dtypes: dict[str, str] = {}
    for name, value in weights.items():
        parameter_name = (
            _matmul_alias(name)
            if token_quantization and value.ndim == 2
            else _conv_alias(name)
            if value.ndim == 2
            else name
        )
        parameter_dtypes[name] = str(ctx.GetParam(parameter_name).dtype)
    int8_parameters = sorted(
        name for name, dtype in parameter_dtypes.items() if "UINT8" in dtype
    )
    executor = None
    if create_executor:
        executor = TFExecutor(
            ctx,
            prefill_executor_config(
                bool(use_hardware),
                software_attn_softmax_impl=not arm_causal,
            ),
        )
    report = {
        **selection,
        "layer": layer_id,
        "seq_len": seq_len,
        "mode": (
            f"quantlite-{attention_mode}-int8-fp16-topk"
            if arm_causal
            else "quantlite-explicit-qdq-int8-fp16-topk"
        ),
        "attention_mode": attention_mode,
        "activation_granularity": activation_granularity,
        "projection_layout": "NCHW-near-square",
        "full_token_grid": list(_factor_token_grid(seq_len)),
        "token_group_grids": {
            name: list(_factor_token_grid(end - begin))
            for name, begin, end in token_group_specs
        },
        "weight_quantization": (
            "QuantizeLite per-channel uint8 in isolated source Q->Conv "
            "islands; final Qwen source graph retains explicit Q/DQ"
        ),
        "token_group_boundaries": list(boundaries),
        "token_groups": [
            {"name": name, "begin": begin, "end": end}
            for name, begin, end in token_group_specs
        ],
        "token_hybrid_scope": (
            "QKV; MLP is FP16 Top-K"
            if split_qkv and fp_mlp
            else "MLP plus QKV"
            if split_qkv
            else "MLP is FP16 Top-K"
            if token_group_specs and fp_mlp
            else "MLP only"
            if token_group_specs
            else "disabled"
        ),
        "token_hybrid_qkv_start_layer": token_hybrid_qkv_start_layer,
        "token_hybrid_qkv_active": split_qkv,
        "fp_attention": fp_attention,
        "fp_mlp": fp_mlp,
        "fp16_parameters": sorted(fp16_parameters),
        "int8_parameter_count": len(int8_parameters),
        "int8_parameters": int8_parameters,
        "parameter_dtypes": parameter_dtypes,
        "dequant_operator": "built-in DeQuantize(dstType=FLOAT16)",
        "source_float_tensors": list(avoid_tensors),
        "quantlite_avoid_tensors": [],
        "residual_dtype": "float16",
        "kv_dtype": "float16",
        "v_attention_path": (
            "UINT8 v_proj -> reshape -> AV"
            if arm_causal and not split_qkv
            else "FP16 group concat -> UINT8 AV"
            if arm_causal
            else "legacy FP16 attention"
        ),
        "v_cache_export_path": (
            "terminal DeQuantize(FP16) from the shared UINT8 V branch"
            if v_cache_export_needs_dequant
            else "existing FP16 V branch"
        ),
        "mlp_compute_path": (
            "UINT8 gate/up -> native Swish -> native Mul -> "
            "UINT8 down_proj"
            if not fp_mlp
            else "FP16 Top-K bypass"
        ),
        "softmax_dtype": "uint8" if arm_causal else "float16",
        "softmax_operator": (
            "ArmCausalMaskSoftmax" if arm_causal else "Softmax"
        ),
        "softmax_threads": int(softmax_threads),
        "causal_mask_input": not arm_causal,
        "post_build_modify": False,
        "qk_qinfo": "H*S" if arm_causal_hxs else "scalar",
        "qk_expanded_rows": expanded_qk_rows,
        "qk_max_multiplier_before": maximum_multiplier_before,
        "qk_max_multiplier_after": maximum_multiplier_after,
        "custom_softmax_input_qinfo": (
            "H*S direct" if arm_causal_hxs else "scalar direct"
        ),
        "score_transport_requant": False,
        "inputs": input_names,
        "outputs": output_names,
        "debug_outputs": selected_output_names,
        "symbols": symbols,
    }
    return ctx, executor, input_names, output_names, report


def manually_quantize_float_layer_graph(
    ctx: Any,
    config: QwenPrefillConfig,
    layer_id: int,
    weights: dict[str, np.ndarray],
    symbols: dict[str, str],
    range_json: str | Path,
    *,
    top_k: int = 0,
    dump_modify_json: str | Path | None = None,
    dump_report: str | Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Turn the exact float graph into INT8/FP16 via ``TFContext.Modify``.

    Unlike ``TFCalibration.Quantize``, this route never asks the SDK to infer
    mixed-precision boundaries.  It installs explicit UINT8 weights and Q/DQ
    nodes only after the valid float graph has closed, matching the stable ViT
    post-quant workflow.
    """
    ranges = json.loads(Path(range_json).read_text())
    selection = select_outlier_branches(ranges, top_k)
    prefix = f"layers.{layer_id}"
    fp_attention = layer_id in selection["fp_attn_layers"]
    fp_mlp = layer_id in selection["fp_mlp_layers"]
    fp16 = "TFDtypeFp16"
    uint8 = "TFDtypeUint8"
    dequant_fp16 = {"param": {"dstType": fp16}}

    def symbol(tag: str) -> str:
        logical = f"{prefix}.{tag}"
        if logical not in symbols:
            raise KeyError(f"symbol map is missing {logical}")
        return symbols[logical]

    def qrange(name: str) -> tuple[list[str], list[str]]:
        return _modify_range(ranges, f"{prefix}.{name}")

    def quant_entry(source: str, name: str, range_name: str) -> dict[str, Any]:
        low, high = qrange(range_name)
        return {
            "input": [source],
            "layerName": name,
            "layerType": "Quantize",
            "output": [name],
            "outputDataType": uint8,
            "OutDataMin": low,
            "OutDataMax": high,
        }

    def dequant_entry(source: str, name: str) -> dict[str, Any]:
        return {
            "input": [source],
            "layerName": name,
            "layerType": "DeQuantize",
            "output": [name],
            **dequant_fp16,
        }

    fp16_parameter_names = {
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
    }
    if fp_attention:
        fp16_parameter_names.add(f"{prefix}.self_attn.o_proj.weight")
    if fp_mlp:
        fp16_parameter_names.update(
            {
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    int8_parameter_names = set(weights) - fp16_parameter_names
    new_parameters: dict[str, np.ndarray] = {}
    int8_param_ranges: dict[str, tuple[float, float]] = {}
    for name in sorted(fp16_parameter_names):
        value = weights[name]
        shaped = (
            value.reshape(value.shape[0], value.shape[1], 1, 1)
            if value.ndim == 2
            else value
        )
        new_parameters[f"PostFP.{name}"] = np.ascontiguousarray(
            shaped, dtype=np.float16
        )
    for name in sorted(int8_parameter_names):
        value = weights[name]
        shaped = (
            value.reshape(value.shape[0], value.shape[1], 1, 1)
            if value.ndim == 2
            else value
        )
        encoded, low, high = _quantize_uint8(shaped)
        parameter_name = f"ManualINT8.{name}"
        new_parameters[parameter_name] = encoded
        int8_param_ranges[parameter_name] = (low, high)
    ctx.RegisterParamToContext(**new_parameters)
    for name, (low, high) in int8_param_ranges.items():
        if not ctx.AddInt8Config(name, high, low):
            raise RuntimeError(f"failed to register INT8 parameter {name}")

    def register(tag: str, source_range: str) -> None:
        low, high = _range_values(ranges, f"{prefix}.{source_range}")
        if not ctx.AddInt8Config(symbol(tag), high, low):
            raise RuntimeError(f"failed to register activation range {tag}")

    activation_ranges = {
        "self_attn.q_proj.conv": "self_attn.q_proj",
        "self_attn.q_proj": "self_attn.q_proj",
        "self_attn.k_proj.conv": "self_attn.k_proj",
        "self_attn.k_proj": "self_attn.k_proj",
        "self_attn.v_proj.conv": "self_attn.v_proj",
        "self_attn.v_proj": "self_attn.v_proj",
        "self_attn.q_rope": "self_attn.q_rope",
        "self_attn.k_rope": "self_attn.k_rope",
        "self_attn.v_heads": "self_attn.v_cache",
        "self_attn.k_repeated": "self_attn.k_rope",
        "self_attn.v_repeated": "self_attn.v_cache",
        "self_attn.q_matmul_input": "self_attn.q_rope",
        "self_attn.k_matmul_input": "self_attn.k_rope",
        "self_attn.v_matmul_input": "self_attn.v_cache",
        "self_attn.qk_matmul": "self_attn.qk_matmul",
        "self_attn.scores": "self_attn.scores",
        "self_attn.attention": "self_attn.attention",
        "self_attn.attention_reshape": "self_attn.attention",
        "self_attn.attention_transposed": "self_attn.attention",
        "self_attn.attention_tokens": "self_attn.attention",
        "self_attn.o_proj.conv": "self_attn.o_proj",
        "self_attn.o_proj": "self_attn.o_proj",
        "mlp.gate_proj.conv": "mlp.gate_proj",
        "mlp.gate_proj": "mlp.gate_proj",
        "mlp.up_proj.conv": "mlp.up_proj",
        "mlp.up_proj": "mlp.up_proj",
        "mlp.gate_silu": "mlp.gate_silu",
        "mlp.mid": "mlp.mid",
        "mlp.down_proj.conv": "mlp.down_proj",
        "mlp.down_proj": "mlp.down_proj",
    }
    for tag, source_range in activation_ranges.items():
        if f"{prefix}.{tag}" in symbols:
            register(tag, source_range)

    hidden = symbol("input")
    input_rms = symbol("input_rms")
    input_norm = symbol("input_norm")
    quant_input = f"PostQuant_QwenInputNorm_{layer_id}"
    layers: list[dict[str, Any]] = [
        {"input": [hidden], "layerName": input_rms, "outputDataType": fp16},
        {
            "input": [input_rms, f"PostFP.{prefix}.input_layernorm.weight"],
            "layerName": input_norm,
            "outputDataType": fp16,
        },
        quant_entry(input_norm, quant_input, "input_norm"),
    ]

    for projection in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        layers.extend(
            (
                {
                    "input": [quant_input],
                    "layerName": symbol(projection + ".transposed"),
                    "outputDataType": uint8,
                },
                {"layerName": symbol(projection + ".conv_input"), "outputDataType": uint8},
                {
                    "input": [
                        symbol(projection + ".conv_input"),
                        f"ManualINT8.{prefix}.{projection}.weight",
                    ],
                    "layerName": symbol(projection + ".conv"),
                    "outputDataType": uint8,
                },
                {"layerName": symbol(projection), "outputDataType": uint8},
            )
        )

    for stem in ("q", "k"):
        projection = f"self_attn.{stem}_proj"
        projection_fp16 = f"PostDeQuant_Qwen{stem.upper()}Proj_{layer_id}"
        quant_norm = f"PostQuant_Qwen{stem.upper()}Norm_{layer_id}"
        layers.extend(
            (
                dequant_entry(symbol(projection), projection_fp16),
                {
                    "input": [projection_fp16],
                    "layerName": symbol(f"self_attn.{stem}_heads_reshape"),
                    "outputDataType": fp16,
                },
                {
                    "input": [symbol(f"self_attn.{stem}_heads_reshape")],
                    "layerName": symbol(f"self_attn.{stem}_rms"),
                    "outputDataType": fp16,
                },
                {
                    "input": [
                        symbol(f"self_attn.{stem}_rms"),
                        f"PostFP.{prefix}.self_attn.{stem}_norm.weight",
                    ],
                    "layerName": symbol(f"self_attn.{stem}_norm"),
                    "outputDataType": fp16,
                },
                quant_entry(
                    symbol(f"self_attn.{stem}_norm"),
                    quant_norm,
                    f"self_attn.{stem}_norm",
                ),
                {
                    "input": [quant_norm],
                    "layerName": symbol(f"self_attn.{stem}_heads"),
                    "outputDataType": uint8,
                },
            )
        )
    layers.extend(
        (
            {"layerName": symbol("self_attn.qk_matmul"), "outputDataType": uint8},
            {"layerName": symbol("self_attn.scores"), "outputDataType": uint8},
        )
    )

    scores_fp16 = f"PostDeQuant_QwenScores_{layer_id}"
    quant_prob = f"PostQuant_QwenProbabilities_{layer_id}"
    layers.extend(
        (
            dequant_entry(symbol("self_attn.scores"), scores_fp16),
            {
                "input": [scores_fp16, symbols["causal_mask"]],
                "layerName": symbol("self_attn.masked_scores"),
                "outputDataType": fp16,
            },
            {"layerName": symbol("self_attn.probabilities"), "outputDataType": fp16},
            quant_entry(
                symbol("self_attn.probabilities"),
                quant_prob,
                "self_attn.probabilities",
            ),
            {
                "input": [quant_prob, symbol("self_attn.v_matmul_input")],
                "layerName": symbol("self_attn.attention"),
                "outputDataType": uint8,
            },
        )
    )

    if fp_attention:
        attention_fp16 = f"PostDeQuant_QwenAttention_{layer_id}"
        layers.append(dequant_entry(symbol("self_attn.attention_tokens"), attention_fp16))
        o_input = attention_fp16
        o_weight = f"PostFP.{prefix}.self_attn.o_proj.weight"
        o_dtype = fp16
    else:
        o_input = symbol("self_attn.attention_tokens")
        o_weight = f"ManualINT8.{prefix}.self_attn.o_proj.weight"
        o_dtype = uint8
    layers.extend(
        (
            {
                "input": [o_input],
                "layerName": symbol("self_attn.o_proj.transposed"),
                "outputDataType": o_dtype,
            },
            {"layerName": symbol("self_attn.o_proj.conv_input"), "outputDataType": o_dtype},
            {
                "input": [symbol("self_attn.o_proj.conv_input"), o_weight],
                "layerName": symbol("self_attn.o_proj.conv"),
                "outputDataType": o_dtype,
            },
            {"layerName": symbol("self_attn.o_proj"), "outputDataType": o_dtype},
        )
    )
    o_fp16 = symbol("self_attn.o_proj")
    if not fp_attention:
        o_fp16 = f"PostDeQuant_QwenOProj_{layer_id}"
        layers.append(dequant_entry(symbol("self_attn.o_proj"), o_fp16))
    residual = symbol("attn_residual")
    post_rms = symbol("post_rms")
    post_norm = symbol("post_norm")
    layers.extend(
        (
            {"input": [hidden, o_fp16], "layerName": residual, "outputDataType": fp16},
            {"input": [residual], "layerName": post_rms, "outputDataType": fp16},
            {
                "input": [post_rms, f"PostFP.{prefix}.post_attention_layernorm.weight"],
                "layerName": post_norm,
                "outputDataType": fp16,
            },
        )
    )

    mlp_dtype = fp16 if fp_mlp else uint8
    mlp_input = post_norm
    if not fp_mlp:
        mlp_input = f"PostQuant_QwenPostNorm_{layer_id}"
        layers.append(quant_entry(post_norm, mlp_input, "post_norm"))
    for projection in ("mlp.gate_proj", "mlp.up_proj"):
        weight_prefix = "PostFP" if fp_mlp else "ManualINT8"
        layers.extend(
            (
                {
                    "input": [mlp_input],
                    "layerName": symbol(projection + ".transposed"),
                    "outputDataType": mlp_dtype,
                },
                {"layerName": symbol(projection + ".conv_input"), "outputDataType": mlp_dtype},
                {
                    "input": [
                        symbol(projection + ".conv_input"),
                        f"{weight_prefix}.{prefix}.{projection}.weight",
                    ],
                    "layerName": symbol(projection + ".conv"),
                    "outputDataType": mlp_dtype,
                },
                {"layerName": symbol(projection), "outputDataType": mlp_dtype},
            )
        )
    layers.extend(
        (
            {"layerName": symbol("mlp.gate_silu"), "outputDataType": mlp_dtype},
            {"layerName": symbol("mlp.mid"), "outputDataType": mlp_dtype},
            {
                "input": [
                    symbol("mlp.down_proj.conv_input"),
                    f"{'PostFP' if fp_mlp else 'ManualINT8'}.{prefix}.mlp.down_proj.weight",
                ],
                "layerName": symbol("mlp.down_proj.conv"),
                "outputDataType": mlp_dtype,
            },
            {"layerName": symbol("mlp.down_proj"), "outputDataType": mlp_dtype},
        )
    )
    down_fp16 = symbol("mlp.down_proj")
    if not fp_mlp:
        down_fp16 = f"PostDeQuant_QwenDownProj_{layer_id}"
        layers.append(dequant_entry(symbol("mlp.down_proj"), down_fp16))
    output = symbol("output")
    k_fp16 = f"qwen_prefill_l{layer_id}_k_fp16"
    v_fp16 = f"qwen_prefill_l{layer_id}_v_fp16"
    layers.extend(
        (
            {"input": [residual, down_fp16], "layerName": output, "outputDataType": fp16},
            dequant_entry(symbol("self_attn.k_rope"), k_fp16),
            dequant_entry(symbol("self_attn.v_heads"), v_fp16),
        )
    )
    modify = {"AddOnPass": [], "DeleteLayer": [], "Layer": layers}
    # Persist the exact rewrite before entering the native SDK.  If Modify
    # rejects an optimized-away node, the request remains available as a
    # standalone reproducer instead of being lost with the exception.
    if dump_modify_json:
        Path(dump_modify_json).write_text(
            json.dumps(modify, indent=2, sort_keys=True)
        )
    ctx.Modify(modify)
    output_names = [output, k_fp16, v_fp16]
    ctx.SetOutputs(output_names)
    report = {
        **selection,
        "layer": layer_id,
        "mode": "manual-postquant-int8-fp16-topk",
        "weight_quantization": "per-tensor asymmetric uint8",
        "fp_attention": fp_attention,
        "fp_mlp": fp_mlp,
        "fp16_parameter_count": len(fp16_parameter_names),
        "int8_parameter_count": len(int8_parameter_names),
        "residual_dtype": "float16",
        "kv_dtype": "float16",
        "softmax_dtype": "float16",
        "outputs": output_names,
        "modified_layer_entries": len(layers),
    }
    if dump_report:
        Path(dump_report).write_text(json.dumps(report, indent=2, sort_keys=True))
    return output_names, report


def _range_values(ranges: dict[str, Any], name: str) -> tuple[float, float]:
    item = ranges.get(name)
    if item is None:
        raise KeyError(f"range JSON is missing {name}")
    low = float(item["min"] if isinstance(item, dict) else item[0])
    high = float(item["max"] if isinstance(item, dict) else item[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"invalid range for {name}: [{low}, {high}]")
    return low, high


def _modify_range(ranges: dict[str, Any], name: str) -> tuple[list[str], list[str]]:
    low, high = _range_values(ranges, name)
    return [repr(low)], [repr(high)]


def apply_int8_fp16_layer_modify(
    ctx: Any,
    config: QwenPrefillConfig,
    layer_id: int,
    weights: dict[str, np.ndarray],
    symbols: dict[str, str],
    ranges: dict[str, Any],
    *,
    top_k: int,
    dump_modify_json: str | Path | None = None,
    dump_bypass_report: str | Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Restore the ViT.py-style FP16 residual and optional Top-K branches."""
    prefix = f"layers.{layer_id}"
    report = select_outlier_branches(ranges, top_k)
    fp_attention = layer_id in report["fp_attn_layers"]
    fp_mlp = layer_id in report["fp_mlp_layers"]
    branch_dtype = "TFDtypeFp16"
    dequant_fp16 = {"param": {"dstType": branch_dtype}}

    def symbol(tag: str) -> str:
        name = f"{prefix}.{tag}" if not tag.startswith("layers.") else tag
        if name not in symbols:
            raise KeyError(f"symbol map is missing {name}")
        return symbols[name]

    norm1_min, norm1_max = _modify_range(ranges, f"{prefix}.input_norm")
    norm2_min, norm2_max = _modify_range(ranges, f"{prefix}.post_norm")
    prob_min, prob_max = _modify_range(
        ranges, f"{prefix}.self_attn.probabilities"
    )
    fp16_limit = float(np.finfo(np.float16).max)
    for name in (
        f"{prefix}.input",
        f"{prefix}.input_norm",
        f"{prefix}.attn_residual",
        f"{prefix}.post_norm",
        f"{prefix}.output",
    ):
        low, high = _range_values(ranges, name)
        if max(abs(low), abs(high)) > fp16_limit:
            raise ValueError(f"FP16 residual range overflow for {name}")

    restored_names = {
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
    }
    if fp_attention:
        restored_names.add(f"{prefix}.self_attn.o_proj.weight")
    if fp_mlp:
        restored_names.update(
            {
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    restored = {
        f"PostFP.{name}": np.ascontiguousarray(weights[name], dtype=np.float16)
        for name in sorted(restored_names)
    }
    ctx.RegisterParamToContext(**restored)

    hidden = symbol("input")
    input_rms = symbol("input_rms")
    input_norm = symbol("input_norm")
    quant_input_norm = f"PostQuant_QwenInputNorm_{layer_id}"
    layers: list[dict[str, Any]] = [
        {"layerName": hidden, "outputDataType": branch_dtype},
        {
            "input": [hidden],
            "layerName": input_rms,
            "outputDataType": branch_dtype,
        },
        {
            "input": [
                input_rms,
                f"PostFP.{prefix}.input_layernorm.weight",
            ],
            "layerName": input_norm,
            "outputDataType": branch_dtype,
        },
        {
            "input": [input_norm],
            "layerName": quant_input_norm,
            "layerType": "Quantize",
            "output": [quant_input_norm],
            "outputDataType": "TFDtypeUint8",
            "OutDataMin": norm1_min,
            "OutDataMax": norm1_max,
        },
    ]
    for projection in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ):
        layers.append(
            {
                "input": [quant_input_norm],
                "layerName": symbol(projection + ".transposed"),
                "outputDataType": "TFDtypeUint8",
            }
        )

    # The causal mask makes the score range unsuitable for INT8.  Keep only
    # masked-score Add + Softmax in FP16 and requantize probabilities for AV.
    score_fp16 = f"PostDeQuant_QwenScores_{layer_id}"
    quant_probabilities = f"PostQuant_QwenProbabilities_{layer_id}"
    layers.extend(
        (
            {
                "input": [symbol("self_attn.scores")],
                "layerName": score_fp16,
                "layerType": "DeQuantize",
                "output": [score_fp16],
                **dequant_fp16,
            },
            {
                "layerName": symbols["causal_mask"],
                "outputDataType": branch_dtype,
            },
            {
                "input": [score_fp16, symbols["causal_mask"]],
                "layerName": symbol("self_attn.masked_scores"),
                "outputDataType": branch_dtype,
            },
            {
                "input": [symbol("self_attn.masked_scores")],
                "layerName": symbol("self_attn.probabilities"),
                "outputDataType": branch_dtype,
            },
            {
                "input": [symbol("self_attn.probabilities")],
                "layerName": quant_probabilities,
                "layerType": "Quantize",
                "output": [quant_probabilities],
                "outputDataType": "TFDtypeUint8",
                "OutDataMin": prob_min,
                "OutDataMax": prob_max,
            },
            {
                "input": [
                    quant_probabilities,
                    symbol("self_attn.v_matmul_input"),
                ],
                "layerName": symbol("self_attn.attention"),
                "outputDataType": "TFDtypeUint8",
            },
        )
    )

    if fp_attention:
        attention_fp16 = f"PostDeQuant_QwenAttention_{layer_id}"
        layers.extend(
            (
                {
                    "input": [symbol("self_attn.attention_tokens")],
                    "layerName": attention_fp16,
                    "layerType": "DeQuantize",
                    "output": [attention_fp16],
                    **dequant_fp16,
                },
                {
                    "input": [attention_fp16],
                    "layerName": symbol("self_attn.o_proj.transposed"),
                    "outputDataType": branch_dtype,
                },
                {
                    "layerName": symbol("self_attn.o_proj.conv_input"),
                    "outputDataType": branch_dtype,
                },
                {
                    "input": [
                        symbol("self_attn.o_proj.conv_input"),
                        f"PostFP.{prefix}.self_attn.o_proj.weight",
                    ],
                    "layerName": symbol("self_attn.o_proj.conv"),
                    "outputDataType": branch_dtype,
                },
                {
                    "layerName": symbol("self_attn.o_proj"),
                    "outputDataType": branch_dtype,
                },
            )
        )
        attention_residual = symbol("self_attn.o_proj")
    else:
        attention_residual = f"PostDeQuant_QwenOProj_{layer_id}"
        layers.append(
            {
                "input": [symbol("self_attn.o_proj")],
                "layerName": attention_residual,
                "layerType": "DeQuantize",
                "output": [attention_residual],
                **dequant_fp16,
            }
        )

    residual = symbol("attn_residual")
    post_rms = symbol("post_rms")
    post_norm = symbol("post_norm")
    layers.extend(
        (
            {
                "input": [hidden, attention_residual],
                "layerName": residual,
                "outputDataType": branch_dtype,
            },
            {
                "input": [residual],
                "layerName": post_rms,
                "outputDataType": branch_dtype,
            },
            {
                "input": [
                    post_rms,
                    f"PostFP.{prefix}.post_attention_layernorm.weight",
                ],
                "layerName": post_norm,
                "outputDataType": branch_dtype,
            },
        )
    )

    if fp_mlp:
        for projection in ("mlp.gate_proj", "mlp.up_proj"):
            layers.extend(
                (
                    {
                        "input": [post_norm],
                        "layerName": symbol(projection + ".transposed"),
                        "outputDataType": branch_dtype,
                    },
                    {
                        "layerName": symbol(projection + ".conv_input"),
                        "outputDataType": branch_dtype,
                    },
                    {
                        "input": [
                            symbol(projection + ".conv_input"),
                            f"PostFP.{prefix}.{projection}.weight",
                        ],
                        "layerName": symbol(projection + ".conv"),
                        "outputDataType": branch_dtype,
                    },
                    {
                        "layerName": symbol(projection),
                        "outputDataType": branch_dtype,
                    },
                )
            )
        layers.extend(
            (
                {
                    "layerName": symbol("mlp.gate_silu"),
                    "outputDataType": branch_dtype,
                },
                {
                    "layerName": symbol("mlp.mid"),
                    "outputDataType": branch_dtype,
                },
                {
                    "input": [symbol("mlp.mid")],
                    "layerName": symbol("mlp.down_proj.transposed"),
                    "outputDataType": branch_dtype,
                },
                {
                    "layerName": symbol("mlp.down_proj.conv_input"),
                    "outputDataType": branch_dtype,
                },
                {
                    "input": [
                        symbol("mlp.down_proj.conv_input"),
                        f"PostFP.{prefix}.mlp.down_proj.weight",
                    ],
                    "layerName": symbol("mlp.down_proj.conv"),
                    "outputDataType": branch_dtype,
                },
                {
                    "layerName": symbol("mlp.down_proj"),
                    "outputDataType": branch_dtype,
                },
            )
        )
        mlp_residual = symbol("mlp.down_proj")
    else:
        quant_post_norm = f"PostQuant_QwenPostNorm_{layer_id}"
        layers.append(
            {
                "input": [post_norm],
                "layerName": quant_post_norm,
                "layerType": "Quantize",
                "output": [quant_post_norm],
                "outputDataType": "TFDtypeUint8",
                "OutDataMin": norm2_min,
                "OutDataMax": norm2_max,
            }
        )
        for projection in ("mlp.gate_proj", "mlp.up_proj"):
            layers.append(
                {
                    "input": [quant_post_norm],
                    "layerName": symbol(projection + ".transposed"),
                    "outputDataType": "TFDtypeUint8",
                }
            )
        mlp_residual = f"PostDeQuant_QwenDownProj_{layer_id}"
        layers.append(
            {
                "input": [symbol("mlp.down_proj")],
                "layerName": mlp_residual,
                "layerType": "DeQuantize",
                "output": [mlp_residual],
                **dequant_fp16,
            }
        )

    output = symbol("output")
    k_fp16 = f"qwen_prefill_l{layer_id}_k_fp16"
    v_fp16 = f"qwen_prefill_l{layer_id}_v_fp16"
    layers.extend(
        (
            {
                "input": [residual, mlp_residual],
                "layerName": output,
                "outputDataType": branch_dtype,
            },
            {
                "input": [symbol("self_attn.k_rope")],
                "layerName": k_fp16,
                "layerType": "DeQuantize",
                "output": [k_fp16],
                **dequant_fp16,
            },
            {
                "input": [symbol("self_attn.v_cache")],
                "layerName": v_fp16,
                "layerType": "DeQuantize",
                "output": [v_fp16],
                **dequant_fp16,
            },
        )
    )
    modify = {"AddOnPass": [], "DeleteLayer": [], "Layer": layers}
    # Dump before the native call so an SDK-side failure remains reproducible.
    if dump_modify_json:
        Path(dump_modify_json).write_text(
            json.dumps(modify, indent=2, sort_keys=True)
        )
    ctx.Modify(modify)
    output_names = [output, k_fp16, v_fp16]
    ctx.SetOutputs(output_names)
    report.update(
        {
            "layer": layer_id,
            "fp_attention": fp_attention,
            "fp_mlp": fp_mlp,
            "residual_dtype": "float16",
            "kv_dtype": "float16",
            "softmax_dtype": "float16",
            "restored_params": sorted(restored),
            "modified_layer_entries": len(layers),
            "outputs": output_names,
        }
    )
    if dump_bypass_report:
        Path(dump_bypass_report).write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )
    return output_names, report


def quantize_layer_graph(
    ctx: Any,
    config: QwenPrefillConfig,
    layer_id: int,
    weights: dict[str, np.ndarray],
    input_names: list[str],
    output_names: list[str],
    symbols: dict[str, str],
    range_json: str | Path,
    output: str | Path,
    *,
    top_k: int = 0,
    dump_modify_json: str | Path | None = None,
    dump_bypass_report: str | Path | None = None,
) -> dict[str, Any]:
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
    # Keep the native PTQ pass topology-only.  Passing broadcast Add/Softmax
    # nodes through avoidtensors crashes this SDK release; the explicit
    # post-quant modification below restores those two nodes to FP16.
    avoid: tuple[str, ...] = ()
    # K/V are interior tensors that are exported only after explicit Q/DQ
    # boundaries are inserted.  Marking them as simultaneous stop-quant
    # outputs makes the SDK quantizer dereference an invalid branch.
    ctx.SetOutputs([output_names[0]])
    calibration.Quantize(
        {
            input_names[0]: TFDataType.TFDL_FLOAT,
            input_names[1]: TFDataType.TFDL_FLOAT,
            input_names[2]: TFDataType.TFDL_FLOAT,
            input_names[3]: TFDataType.TFDL_FLOAT,
        },
        avoidtensors=avoid,
        stopquanttensors=(output_names[0],),
        MergeConcate=False,
        Perchannel=True,
    )
    ranges = json.loads(Path(range_json).read_text())
    new_outputs, report = apply_int8_fp16_layer_modify(
        ctx,
        config,
        layer_id,
        weights,
        symbols,
        ranges,
        top_k=top_k,
        dump_modify_json=dump_modify_json,
        dump_bypass_report=dump_bypass_report,
    )
    ctx.SetOutputs(new_outputs)
    dump_context(ctx, output)
    return report


def execute_layer(
    executor: Any,
    hidden: np.ndarray,
    sin: np.ndarray,
    cos: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = executor.GetInputs()
    if len(inputs) == 3:
        values = (hidden, sin, cos)
    elif len(inputs) == 4:
        if mask is None:
            mask = causal_mask(int(hidden.shape[1]))
        values = (hidden, sin, cos, mask)
    else:
        raise RuntimeError(
            f"executor has {len(inputs)} inputs, expected 3 or 4"
        )
    from TFDL2.Common import TFDataType

    for tensor, value in zip(inputs, values):
        array = np.ascontiguousarray(value)
        if tensor.dtype == TFDataType.TFDL_FLOAT16:
            array = np.ascontiguousarray(array, dtype=np.float16)
        elif tensor.dtype == TFDataType.TFDL_FLOAT:
            array = np.ascontiguousarray(array, dtype=np.float32)
        elif tensor.dtype == TFDataType.TFDL_UINT8 and array.dtype != np.uint8:
            qmin = float(tensor.qmin[0])
            qmax = float(tensor.qmax[0])
            scale = 255.0 / (qmax - qmin)
            array = np.clip(np.rint((array - qmin) * scale), 0, 255).astype(
                np.uint8
            )
        tensor.fromNumpy(array)
    outputs = executor()
    if len(outputs) != 3:
        raise RuntimeError(f"executor has {len(outputs)} outputs, expected 3")
    return tuple(output.toNumpy() for output in outputs)  # type: ignore[return-value]


def merge_range_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        raw = json.loads(Path(path).read_text())
        overlap = set(merged).intersection(raw)
        if overlap:
            raise ValueError(f"duplicate range entries: {sorted(overlap)[:5]}")
        merged.update(raw)
    return merged


def select_outlier_branches(
    ranges: dict[str, Any], top_k: int
) -> dict[str, Any]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    candidates: list[dict[str, Any]] = []
    for name, item in ranges.items():
        if not (name.endswith(".self_attn.o_proj") or name.endswith(".mlp.down_proj")):
            continue
        low = float(item["min"] if isinstance(item, dict) else item[0])
        high = float(item["max"] if isinstance(item, dict) else item[1])
        candidates.append(
            {
                "name": name,
                "abs_range": max(abs(low), abs(high)),
                "kind": "attention" if ".self_attn." in name else "mlp",
                "layer": int(name.split(".")[1]),
            }
        )
    candidates.sort(key=lambda item: float(item["abs_range"]), reverse=True)
    selected = candidates[: min(top_k, len(candidates))]
    return {
        "top_k": int(top_k),
        "selected": selected,
        "fp_attn_layers": sorted(
            {int(item["layer"]) for item in selected if item["kind"] == "attention"}
        ),
        "fp_mlp_layers": sorted(
            {int(item["layer"]) for item in selected if item["kind"] == "mlp"}
        ),
        "ranked_candidates": candidates,
    }
