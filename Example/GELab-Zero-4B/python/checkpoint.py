#!/usr/bin/env python3
"""Low-memory Qwen3-VL checkpoint inspection and tensor access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ModelContract:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: tuple[int, int, int]
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    vision_hidden_size: int
    vision_intermediate_size: int
    vision_depth: int
    vision_heads: int
    vision_patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    vision_out_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...]
    num_position_embeddings: int

    @classmethod
    def from_model(cls, model_path: str | Path) -> "ModelContract":
        root = Path(model_path)
        raw = json.loads((root / "config.json").read_text())
        text = raw["text_config"]
        vision = raw["vision_config"]
        rope = text.get("rope_scaling") or {}
        result = cls(
            model_type=str(raw["model_type"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])),
            vocab_size=int(text["vocab_size"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text.get("rope_theta", 5_000_000.0)),
            mrope_section=tuple(int(v) for v in rope.get("mrope_section", (24, 20, 20))),
            image_token_id=int(raw["image_token_id"]),
            video_token_id=int(raw["video_token_id"]),
            vision_start_token_id=int(raw["vision_start_token_id"]),
            vision_end_token_id=int(raw["vision_end_token_id"]),
            vision_hidden_size=int(vision["hidden_size"]),
            vision_intermediate_size=int(vision["intermediate_size"]),
            vision_depth=int(vision["depth"]),
            vision_heads=int(vision["num_heads"]),
            vision_patch_size=int(vision["patch_size"]),
            temporal_patch_size=int(vision["temporal_patch_size"]),
            spatial_merge_size=int(vision["spatial_merge_size"]),
            vision_out_hidden_size=int(vision["out_hidden_size"]),
            deepstack_visual_indexes=tuple(int(v) for v in vision.get("deepstack_visual_indexes", ())),
            num_position_embeddings=int(vision["num_position_embeddings"]),
        )
        result.validate()
        return result

    @property
    def vision_head_dim(self) -> int:
        return self.vision_hidden_size // self.vision_heads

    @property
    def patch_vector_size(self) -> int:
        return 3 * self.temporal_patch_size * self.vision_patch_size**2

    @property
    def kv_repeats(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def query_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    def validate(self) -> None:
        if self.model_type != "qwen3_vl":
            raise ValueError(f"expected model_type='qwen3_vl', got {self.model_type!r}")
        if self.hidden_size != self.vision_out_hidden_size:
            raise ValueError("vision output width and text hidden width differ")
        if self.head_dim <= 0 or self.num_attention_heads <= 0:
            raise ValueError("invalid text attention geometry")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("GQA head count is not divisible")
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError("invalid vision attention geometry")
        if self.spatial_merge_size <= 0 or self.temporal_patch_size <= 0:
            raise ValueError("patch merge factors must be positive")
        if len(self.mrope_section) != 3 or sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(
                "mrope_section must contain three parts summing to head_dim/2"
            )
        if tuple(self.deepstack_visual_indexes) != tuple(sorted(self.deepstack_visual_indexes)):
            raise ValueError("deepstack_visual_indexes must be ordered")

    def as_dict(self) -> dict[str, object]:
        value = dict(self.__dict__)
        value["mrope_section"] = list(self.mrope_section)
        value["deepstack_visual_indexes"] = list(self.deepstack_visual_indexes)
        value["vision_head_dim"] = self.vision_head_dim
        value["patch_vector_size"] = self.patch_vector_size
        value["query_size"] = self.query_size
        return value


class SafeTensorIndex:
    """Read only requested safetensors without materializing the 17.5 GB model."""

    def __init__(self, model_path: str | Path) -> None:
        self.root = Path(model_path)
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            self.weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            single = self.root / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(f"no safetensors checkpoint under {self.root}")
            with single.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                header = json.loads(stream.read(header_size))
            self.weight_map = {
                name: single.name for name in header if name != "__metadata__"
            }

    @staticmethod
    def _decode(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        if dtype == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            value = (bits << 16).view(np.float32)
        elif dtype == "F16":
            value = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dtype == "F32":
            value = np.frombuffer(raw, dtype="<f4")
        elif dtype == "I8":
            value = np.frombuffer(raw, dtype=np.int8)
        else:
            raise TypeError(f"unsupported safetensors dtype {dtype}")
        return np.ascontiguousarray(value.reshape(shape))

    def first(self, candidates: Iterable[str]) -> str | None:
        return next((name for name in candidates if name in self.weight_map), None)

    def read(self, names: Iterable[str]) -> dict[str, np.ndarray]:
        requested = list(dict.fromkeys(names))
        by_shard: dict[str, list[str]] = {}
        for name in requested:
            shard = self.weight_map.get(name)
            if shard is None:
                raise KeyError(f"checkpoint is missing {name}")
            by_shard.setdefault(shard, []).append(name)
        result: dict[str, np.ndarray] = {}
        for shard, shard_names in sorted(by_shard.items()):
            with (self.root / shard).open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                header = json.loads(stream.read(header_size))
                base = 8 + header_size
                for name in shard_names:
                    meta = header[name]
                    start, end = (int(v) for v in meta["data_offsets"])
                    stream.seek(base + start)
                    result[name] = self._decode(
                        stream.read(end - start),
                        str(meta["dtype"]),
                        tuple(int(v) for v in meta["shape"]),
                    )
        return result

    def read_rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"checkpoint is missing {name}")
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        with (self.root / shard).open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_size))
            meta = header[name]
            shape = tuple(int(v) for v in meta["shape"])
            if len(shape) != 2:
                raise ValueError(f"{name} is not a matrix: {shape}")
            if np.any(rows < 0) or np.any(rows >= shape[0]):
                raise IndexError(f"row outside {name} shape {shape}")
            dtype = str(meta["dtype"])
            item_size = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
            if item_size is None:
                raise TypeError(f"unsupported row dtype {dtype}")
            row_bytes = shape[1] * item_size
            base = 8 + header_size + int(meta["data_offsets"][0])
            output = np.empty((rows.size, shape[1]), dtype=np.float32)
            for target, source in enumerate(rows.tolist()):
                stream.seek(base + source * row_bytes)
                output[target] = self._decode(
                    stream.read(row_bytes), dtype, (shape[1],)
                )
        return output


def embedding_weight_name(index: SafeTensorIndex) -> str:
    name = index.first(
        (
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
    )
    if name is None:
        raise KeyError("checkpoint has no text embedding table")
    return name


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
