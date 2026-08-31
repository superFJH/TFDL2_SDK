#!/usr/bin/env python3
"""Read the small non-FB artifacts required by a checkpoint-free deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


RUNTIME_ASSETS_FORMAT = "gelab-zero-4b-runtime-assets-v1"


class RuntimeEmbeddingReader:
    """Random-access reader for the exported BF16 token embedding table."""

    def __init__(self, assets_dir: str | Path) -> None:
        self.root = Path(assets_dir).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing runtime-assets manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != RUNTIME_ASSETS_FORMAT:
            raise ValueError("unsupported GELab runtime-assets format")
        embedding = manifest.get("token_embedding")
        if not isinstance(embedding, dict):
            raise ValueError("runtime-assets manifest has no token_embedding")
        if embedding.get("dtype") != "BF16":
            raise ValueError("runtime token embedding must preserve BF16 values")
        shape = tuple(int(value) for value in embedding.get("shape", []))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"invalid token embedding shape: {shape}")
        self.shape = shape
        self.path = self.root / str(embedding.get("file", ""))
        expected_bytes = int(np.prod(shape)) * 2
        if not self.path.is_file() or self.path.stat().st_size != expected_bytes:
            raise FileNotFoundError(
                f"invalid BF16 token embedding: {self.path} "
                f"(expected {expected_bytes} bytes)"
            )
        self._rows = np.memmap(self.path, mode="r", dtype="<u2", shape=shape)
        self.name = str(embedding.get("source_tensor", "token_embedding"))
        final_norm = manifest.get("final_norm")
        self._final_norm: np.ndarray | None = None
        if isinstance(final_norm, dict):
            norm_shape = tuple(int(value) for value in final_norm.get("shape", []))
            norm_path = self.root / str(final_norm.get("file", ""))
            expected_norm_bytes = self.shape[1] * 2
            if (
                final_norm.get("dtype") != "BF16"
                or norm_shape != (self.shape[1],)
                or not norm_path.is_file()
                or norm_path.stat().st_size != expected_norm_bytes
            ):
                raise ValueError("runtime final RMSNorm asset is invalid")
            bits = np.fromfile(norm_path, dtype="<u2")
            self._final_norm = np.ascontiguousarray(
                (bits.astype(np.uint32) << 16).view(np.float32)
            )
        raw_config = json.loads((self.root / "config.json").read_text())
        text_config = raw_config.get("text_config", raw_config)
        self.rms_norm_eps = float(text_config.get("rms_norm_eps", 1e-6))

    def read_rows(self, rows: np.ndarray) -> np.ndarray:
        indices = np.asarray(rows, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise IndexError("token embedding row is outside the vocabulary")
        # BF16 is stored exactly as the checkpoint's little-endian uint16 bit
        # pattern. Convert only requested rows to FP32 at the API boundary.
        bits = np.asarray(self._rows[indices], dtype=np.uint16)
        return np.ascontiguousarray(
            (bits.astype(np.uint32) << 16).view(np.float32), dtype=np.float32
        )

    def get(self, token: int) -> np.ndarray:
        return np.ascontiguousarray(
            self.read_rows(np.asarray([token], dtype=np.int64)).reshape(
                1, 1, self.shape[1]
            ),
            dtype=np.float32,
        )

    def tied_output_logits(
        self, hidden: np.ndarray, *, block_rows: int = 4096
    ) -> np.ndarray:
        """Apply GELab's tied BF16 token embedding as the output projection.

        The checkpoint explicitly enables ``tie_word_embeddings``.  The
        final RMSNorm is applied before the blocked LM-head GEMV, matching the
        former ONNX final-head graph without materialising a FP32 vocabulary
        matrix. ``hidden`` is only the final prefill token, so the traversal
        is deterministic and request-local.
        """
        value = np.ascontiguousarray(hidden, dtype=np.float32)
        if value.shape != (1, 1, self.shape[1]):
            raise ValueError(
                f"final hidden shape {value.shape}, expected [1,1,{self.shape[1]}]"
            )
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        if self._final_norm is None:
            raise ValueError(
                "runtime assets have no final_norm; re-export runtime assets for "
                "the llama.cpp external-KV decoder"
            )
        vector = value.reshape(self.shape[1])
        vector = vector * np.reciprocal(
            np.sqrt(np.mean(vector * vector, dtype=np.float32) + self.rms_norm_eps)
        ) * self._final_norm
        vocab_size = self.shape[0]
        logits = np.empty(vocab_size, dtype=np.float32)
        for start in range(0, vocab_size, block_rows):
            end = min(vocab_size, start + block_rows)
            weights = self.read_rows(np.arange(start, end, dtype=np.int64))
            logits[start:end] = weights @ vector
        return np.ascontiguousarray(logits.reshape(1, 1, vocab_size))
