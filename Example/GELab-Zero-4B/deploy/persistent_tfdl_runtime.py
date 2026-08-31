#!/usr/bin/env python3
"""Process-resident TFDL runtime used by the GELab HTTP service.

The command-line deployment runner intentionally creates short-lived TFDL
processes because that is convenient for exporting and debugging.  A server
must not do that: NPU graph construction then becomes part of the first user
request.  This module owns the selected vision FB plus all fixed-S prefill FBs
for the lifetime of the HTTP process.
"""

from __future__ import annotations

import codecs
from copy import deepcopy
from dataclasses import dataclass
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from project_compat import prepend_shared_paths  # noqa: E402

prepend_shared_paths()

from checkpoint import ModelContract  # noqa: E402
from contract import (  # noqa: E402
    inject_deepstack,
    multimodal_position_ids,
    text_mrope,
    vision_rope,
)
import qwen_prefill as prefill  # noqa: E402
from runtime_assets import RuntimeEmbeddingReader  # noqa: E402


PREFILL_FORMAT = "mage-qwen-prefill-kv-v1"


def _gpt2_byte_decoder() -> dict[str, int]:
    """Return Qwen2's byte-level BPE unicode-symbol to byte mapping."""
    values = list(range(ord("!"), ord("~") + 1)) + list(
        range(ord("¡"), ord("¬") + 1)
    ) + list(range(ord("®"), ord("ÿ") + 1))
    symbols = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            symbols.append(256 + extra)
            extra += 1
    return {chr(symbol): byte for byte, symbol in zip(values, symbols)}


class TokenDeltaDecoder:
    """Decode byte-level BPE token IDs into append-only UTF-8 text chunks.

    A full tokenizer decode can replace an incomplete UTF-8 suffix emitted by
    the previous token. Sending that complete replacement as an OpenAI SSE
    delta duplicates text because the wire protocol has no replacement event.
    Feeding the underlying bytes to an incremental UTF-8 decoder holds
    incomplete characters until they are complete, so every returned fragment
    is safe to append.
    """

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        self.byte_decoder = _gpt2_byte_decoder()
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.special_ids = {
            int(value) for value in getattr(tokenizer, "all_special_ids", [])
        }
        self.text = ""

    def append(self, token_id: int) -> str:
        if int(token_id) in self.special_ids:
            return ""
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if not isinstance(token, str):
            raise TypeError(f"tokenizer returned non-string token for {token_id}")
        try:
            raw = bytes(self.byte_decoder[character] for character in token)
        except KeyError as error:
            raise ValueError(
                f"token {token_id} is not compatible with Qwen byte-level streaming"
            ) from error
        delta = self.decoder.decode(raw, final=False)
        self.text += delta
        return delta

    def finish(self) -> str:
        delta = self.decoder.decode(b"", final=True)
        self.text += delta
        return delta


def _feed_vision_input(tensor: object, value: np.ndarray) -> None:
    dtype = str(getattr(tensor, "dtype"))
    if "FLOAT16" in dtype:
        value = value.astype(np.float16)
    elif "FLOAT" in dtype:
        value = value.astype(np.float32)
    else:
        raise TypeError(f"vision graph cannot consume runtime input dtype {dtype}")
    tensor.fromNumpy(np.ascontiguousarray(value))


class SingleVisionExecutor:
    """Small local copy of the one-FB ABI wrapper, independent of sys.path."""

    def __init__(self, fb_path: Path, executor_config: dict[str, object]) -> None:
        from TFDL2 import TFContext, TFExecutor

        self.path = Path(fb_path)
        self.context = TFContext(path=str(self.path))
        self.executor = TFExecutor(self.context, executor_config)

    def __call__(
        self, pixels: np.ndarray, sin: np.ndarray, cos: np.ndarray
    ) -> list[np.ndarray]:
        inputs = self.executor.GetInputs()
        if len(inputs) == 1:
            values = (pixels,)
        elif len(inputs) == 3:
            values = (pixels, sin, cos)
        else:
            raise ValueError(
                f"single vision graph has {len(inputs)} inputs, expected 1 or 3"
            )
        for tensor, value in zip(inputs, values):
            _feed_vision_input(tensor, value)
        return [tensor.toNumpy().astype(np.float32) for tensor in self.executor()]


def _topology_file(root: Path, manifest: dict[str, object]) -> Path:
    candidates = [
        root / str(item["name"])
        for item in manifest["files"]
        if str(item["name"]).endswith(".fb")
        and not str(item["name"]).endswith(".param.fb")
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one topology FB, got {candidates}")
    return candidates[0]


class ResidentGelabInputBuilder:
    """Keep GELab image preprocessing, tokenizer, and embedding lookup resident."""

    def __init__(
        self,
        *,
        model_path: Path,
        seq_len: int,
        tokenizer: object,
        embeddings: RuntimeEmbeddingReader,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.seq_len = int(seq_len)
        self.contract = ModelContract.from_model(self.model_path)
        if embeddings.shape != (self.contract.vocab_size, self.contract.hidden_size):
            raise ValueError(
                "runtime embedding shape does not match GELab model contract: "
                f"{embeddings.shape}"
            )
        from transformers import AutoProcessor
        from prepare_media import _letterbox_rgb, _load_image

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self._letterbox_rgb = _letterbox_rgb
        self._load_image = _load_image
        self.pad_token_id = int(
            getattr(tokenizer, "pad_token_id", None)
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(tokenizer, "eos_token_id")
        )
        self.startup = {
            "initialized": True,
            "processor": type(self.processor).__name__,
            "input_transport": "process-resident-memory",
            "seq_len": self.seq_len,
        }

    @staticmethod
    def _numpy(value: object, dtype: np.dtype) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def prepare_image(
        self,
        image_path: Path,
        *,
        question: str,
        system: str,
        height: int,
        width: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build one deployment-identical image bundle without temporary files."""
        if height <= 0 or width <= 0:
            raise ValueError("image bucket dimensions must be positive")
        factor = self.contract.vision_patch_size * self.contract.spatial_merge_size
        if height % factor or width % factor:
            raise ValueError(f"image bucket must be divisible by {factor}")
        source, source_info = self._load_image(Path(image_path))
        bucketed = self._letterbox_rgb(source, height, width)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processed = self.processor(
            images=[bucketed],
            text=[prompt],
            do_resize=False,
            return_tensors="pt",
        )
        pixels = self._numpy(processed["pixel_values"], np.dtype(np.float32))
        grid = self._numpy(processed["image_grid_thw"], np.dtype(np.int64))
        input_ids = self._numpy(processed["input_ids"], np.dtype(np.int64))
        attention_mask = self._numpy(processed["attention_mask"], np.dtype(np.int64))
        expected_grid = np.asarray(
            [[1, height // self.contract.vision_patch_size, width // self.contract.vision_patch_size]],
            dtype=np.int64,
        )
        if grid.shape != (1, 3) or not np.array_equal(grid, expected_grid):
            raise ValueError(
                f"processor grid {grid.tolist()} does not match fixed bucket "
                f"{expected_grid.tolist()}"
            )
        expected_pixels = int(np.prod(expected_grid)) * self.contract.patch_vector_size
        if pixels.size != expected_pixels:
            raise ValueError(
                f"processor pixel values have {pixels.size} entries, expected {expected_pixels}"
            )
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
            raise ValueError("processor IDs/mask must have matching [1,S] shapes")
        valid_sequence = int(attention_mask.sum())
        if valid_sequence != input_ids.shape[1] or not np.all(attention_mask == 1):
            raise ValueError("resident processor expects one unpadded prompt")
        bundle = {
            "pixels": np.ascontiguousarray(pixels),
            "grid_thw": np.ascontiguousarray(grid),
            "input_ids_valid": np.ascontiguousarray(input_ids),
            "attention_mask_valid": np.ascontiguousarray(attention_mask),
            "kind": "image",
        }
        report = {
            "input_transport": "process-resident-memory",
            "kind": "image",
            "media": str(Path(image_path).resolve()),
            "question": question,
            "system": system,
            "prompt": prompt,
            "bucket_height": height,
            "bucket_width": width,
            "grid_thw": grid.tolist(),
            "patches": int(np.prod(grid)),
            "visual_tokens": int(np.prod(grid) // self.contract.spatial_merge_size**2),
            "valid_seq_len": valid_sequence,
            "media_info": source_info,
        }
        return bundle, report

    def prepare_external_visual_prompt(
        self,
        *,
        question: str,
        system: str,
        height: int,
        width: int,
        visual_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build an image prompt for trusted externally supplied vision tokens.

        A synthetic fixed-bucket image is given only to the official processor
        so it expands the image placeholder into the exact token sequence. Its
        pixels are never passed to the visual encoder or retained.
        """
        if height <= 0 or width <= 0:
            raise ValueError("image bucket dimensions must be positive")
        factor = self.contract.vision_patch_size * self.contract.spatial_merge_size
        if height % factor or width % factor:
            raise ValueError(f"image bucket must be divisible by {factor}")
        expected_grid = np.asarray(
            [[1, height // self.contract.vision_patch_size, width // self.contract.vision_patch_size]],
            dtype=np.int64,
        )
        expected_visual_tokens = int(
            np.prod(expected_grid) // self.contract.spatial_merge_size**2
        )
        if visual_tokens != expected_visual_tokens:
            raise ValueError(
                f"external features contain {visual_tokens} visual tokens; "
                f"fixed {height}x{width} bucket requires {expected_visual_tokens}"
            )
        from PIL import Image

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "external://vision-features"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processed = self.processor(
            images=[Image.new("RGB", (width, height))],
            text=[prompt],
            do_resize=False,
            return_tensors="pt",
        )
        grid = self._numpy(processed["image_grid_thw"], np.dtype(np.int64))
        input_ids = self._numpy(processed["input_ids"], np.dtype(np.int64))
        attention_mask = self._numpy(processed["attention_mask"], np.dtype(np.int64))
        if grid.shape != (1, 3) or not np.array_equal(grid, expected_grid):
            raise ValueError(
                f"processor grid {grid.tolist()} does not match fixed bucket "
                f"{expected_grid.tolist()}"
            )
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
            raise ValueError("processor IDs/mask must have matching [1,S] shapes")
        valid_sequence = int(attention_mask.sum())
        if valid_sequence != input_ids.shape[1] or not np.all(attention_mask == 1):
            raise ValueError("resident processor expects one unpadded prompt")
        bundle = {
            "grid_thw": np.ascontiguousarray(grid),
            "input_ids_valid": np.ascontiguousarray(input_ids),
            "attention_mask_valid": np.ascontiguousarray(attention_mask),
            "kind": "image",
        }
        report = {
            "input_transport": "external-vision-features-memory",
            "kind": "image",
            "question": question,
            "system": system,
            "prompt": prompt,
            "bucket_height": height,
            "bucket_width": width,
            "grid_thw": grid.tolist(),
            "visual_tokens": visual_tokens,
            "valid_seq_len": valid_sequence,
        }
        return bundle, report

    def build_prefill(
        self, media: dict[str, Any], visual: dict[str, np.ndarray]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Inject resident vision tensors into the fixed-S prefill input."""
        input_ids_valid = np.asarray(media["input_ids_valid"], dtype=np.int64)
        mask_valid = np.asarray(media["attention_mask_valid"], dtype=np.int64)
        if input_ids_valid.shape != mask_valid.shape or input_ids_valid.shape[0] != 1:
            raise ValueError("resident media IDs/mask must have matching [1,S] shapes")
        original_valid = int(mask_valid.sum())
        if original_valid != input_ids_valid.shape[1] or not np.all(mask_valid == 1):
            raise ValueError("resident media prompt must be unpadded")
        if original_valid > self.seq_len:
            raise ValueError(
                f"prompt needs {original_valid} tokens; fixed prefill bucket is {self.seq_len}"
            )
        input_ids = np.full((1, self.seq_len), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((1, self.seq_len), dtype=np.int64)
        input_ids[:, :original_valid] = input_ids_valid
        attention_mask[:, :original_valid] = 1
        required = ("main", "deepstack_0", "deepstack_1", "deepstack_2")
        missing = [name for name in required if name not in visual]
        if missing:
            raise KeyError(f"resident vision values are missing {missing}")
        main_visual = np.ascontiguousarray(visual["main"], dtype=np.float32)
        deepstack = np.stack(
            [np.asarray(visual[f"deepstack_{index}"], dtype=np.float32) for index in range(3)]
        )
        if main_visual.ndim != 2 or main_visual.shape[1] != self.contract.hidden_size:
            raise ValueError(f"resident main vision tensor has invalid shape {main_visual.shape}")
        if deepstack.shape != (3,) + main_visual.shape:
            raise ValueError(
                f"resident DeepStack shape {deepstack.shape}, expected {(3,) + main_visual.shape}"
            )
        visual_mask = (input_ids == self.contract.image_token_id) & (attention_mask == 1)
        if int(visual_mask.sum()) != main_visual.shape[0]:
            raise ValueError(
                f"prompt has {int(visual_mask.sum())} visual tokens, encoder produced "
                f"{main_visual.shape[0]}"
            )
        unique, inverse = np.unique(input_ids.reshape(-1), return_inverse=True)
        hidden = self.embeddings.read_rows(unique)[inverse].reshape(
            1, self.seq_len, self.contract.hidden_size
        )
        hidden[visual_mask] = main_visual
        grid = np.asarray(media["grid_thw"], dtype=np.int64)
        position_ids_3d, rope_delta = multimodal_position_ids(
            input_ids,
            attention_mask,
            self.contract,
            image_grid_thw=grid,
        )
        sin, cos = text_mrope(
            position_ids_3d,
            self.contract.head_dim,
            self.contract.rope_theta,
            self.contract.mrope_section,
        )
        values = {
            "hidden": np.ascontiguousarray(hidden, dtype=np.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": np.ascontiguousarray(position_ids_3d[0], dtype=np.int64),
            "rope_sin": np.ascontiguousarray(sin, dtype=np.float32),
            "rope_cos": np.ascontiguousarray(cos, dtype=np.float32),
            "visual_mask": visual_mask,
            "deepstack": np.ascontiguousarray(deepstack, dtype=np.float32),
            "valid_seq_len": original_valid,
        }
        report = {
            "input_transport": "process-resident-memory",
            "seq_len": self.seq_len,
            "model_seq_len": self.seq_len,
            "valid_seq_len": original_valid,
            "hidden_size": self.contract.hidden_size,
            "visual_tokens": int(main_visual.shape[0]),
            "rope_delta": int(rope_delta[0, 0]),
        }
        return values, report


class RequestKvCache:
    """One request's FP16 KV cache in compact llama.cpp-importable storage."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        prompt_length: int,
        capacity: int,
    ) -> None:
        if not 0 < prompt_length <= capacity:
            raise ValueError("invalid request KV length/capacity")
        self.length = int(prompt_length)
        self.capacity = int(capacity)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        shape = (1, self.capacity, self.num_kv_heads, self.head_dim)
        self.keys = [np.empty(shape, dtype=np.float16) for _ in range(self.num_layers)]
        self.values = [np.empty(shape, dtype=np.float16) for _ in range(self.num_layers)]

    @property
    def nbytes(self) -> int:
        return sum(item.nbytes for item in self.keys) + sum(
            item.nbytes for item in self.values
        )

    def set_prefill(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"invalid KV layer {layer}")
        expected = (1, self.num_kv_heads, self.length, self.head_dim)
        key = np.asarray(key)
        value = np.asarray(value)
        if key.dtype != np.float16 or value.dtype != np.float16:
            raise TypeError(
                f"prefill layer {layer} KV must be FP16, got {key.dtype}/{value.dtype}"
            )
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                f"prefill layer {layer} KV shape {key.shape}/{value.shape}, expected {expected}"
            )
        self.keys[layer][:, : self.length] = key.transpose(0, 2, 1, 3)
        self.values[layer][:, : self.length] = value.transpose(0, 2, 1, 3)

@dataclass
class ResidentPrefillState:
    """Ephemeral prefill outputs needed only by the current decode loop."""

    manifest: dict[str, Any]
    cache: RequestKvCache
    initial_logits: np.ndarray


class PersistentLlamaCppDecoder:
    """Persistent llama.cpp decoder fed by the request's TFDL prefill KV.

    The GGUF model is loaded once in a native worker at API startup.  A TFDL
    prefill graph has a fixed physical bucket, but ``valid_sequence`` changes
    per request.  This class is the only hand-off point and serializes exactly
    that valid KV prefix, never the uninitialised right-padding capacity.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        decoder_dir: Path,
        threads: int,
        kv_cache_type: str = "fp16",
        binary: Path | None = None,
    ) -> None:
        if threads < 0:
            raise ValueError("llama.cpp decode threads must be non-negative")
        self.model_path = Path(model_path).resolve()
        self.decoder_dir = Path(decoder_dir).resolve()
        self.config = prefill.QwenPrefillConfig.from_model(self.model_path)
        self.manifest = PersistentGelabTFDLRuntime._read_manifest(
            self.decoder_dir / "manifest.json"
        )
        if self.manifest.get("format") != "tfdl-llmdecode-gguf-v1":
            raise ValueError("unsupported llama.cpp decoder manifest")
        config = self.manifest.get("config", {})
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        ):
            if int(config.get(name, -1)) != int(getattr(self.config, name)):
                raise ValueError(f"llama.cpp decoder/model config mismatch at {name}")
        if not bool(self.manifest.get("tie_word_embeddings", False)):
            raise ValueError("checkpoint-free final head requires tie_word_embeddings=true")
        from transformers import AutoTokenizer
        from tfdl_llmdecode import ExternalKvDecodeWorker

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.eos_token = int(getattr(self.tokenizer, "eos_token_id"))
        self.embeddings = RuntimeEmbeddingReader(self.model_path)
        model_file = self.decoder_dir / str(self.manifest.get("model_file", ""))
        self.threads = int(threads)
        self.kv_cache_type = str(kv_cache_type)
        self.worker = ExternalKvDecodeWorker(
            model=model_file,
            threads=self.threads,
            kv_cache_type=self.kv_cache_type,
            binary=binary,
        )
        self.startup = {
            "initialized": True,
            "engine": "llama.cpp-external-kv",
            "model": str(model_file),
            "worker_pid": self.worker.pid,
            "worker_model_load_seconds": self.worker.startup_seconds,
            "threads": self.threads,
            "kv_cache_type": self.kv_cache_type,
            "final_head": "runtime-assets tied BF16 token embedding",
        }

    def close(self) -> None:
        self.worker.close()

    def initial_logits(self, hidden: np.ndarray) -> np.ndarray:
        """Calculate just the seed logits; all decoder layers run in GGUF."""
        return self.embeddings.tied_output_logits(hidden)

    @staticmethod
    def _validate_prompt(
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        valid_sequence: int,
    ) -> int:
        if input_ids.shape != attention_mask.shape:
            raise ValueError("prompt input IDs and mask must have identical shapes")
        if input_ids.shape[0] != 1:
            raise ValueError("llama.cpp external KV decoder supports batch size one")
        model_sequence = int(input_ids.shape[1])
        if not 0 < valid_sequence <= model_sequence:
            raise ValueError("invalid prompt valid_seq_len")
        expected_mask = np.zeros_like(attention_mask)
        expected_mask[:, :valid_sequence] = 1
        if not np.array_equal(attention_mask, expected_mask):
            raise ValueError("prompt must use one valid prefix followed by right-padding")
        position_ids = np.asarray(position_ids)
        if position_ids.shape not in ((3, model_sequence), (1, model_sequence)):
            raise ValueError(
                "MRoPE position IDs must be [3,S] for resident input or the "
                f"legacy first-axis [1,S], got {position_ids.shape}"
            )
        return model_sequence

    def _report(
        self,
        *,
        native: dict[str, Any],
        model_sequence: int,
        cache_seconds: float,
        cache_bytes: int,
        text: str,
        output_json: Path,
    ) -> dict[str, Any]:
        tokens = [int(value) for value in native["generated_token_ids"]]
        report = {
            "format": "gelab-llama.cpp-external-kv-decode-v1",
            "model_path": str(self.model_path),
            "decoder": self.startup["model"],
            "prompt_dir": "in-memory",
            "prefill_dir": "in-memory",
            "prompt_tokens": int(native["prompt_tokens"]),
            "model_seq_len": model_sequence,
            "generated_tokens": len(tokens),
            "npu_seed_token": tokens[0] if tokens else None,
            "cpu_decode_steps": int(native["decode_calls"]),
            "threads": self.threads,
            "cache_load_seconds": cache_seconds,
            "cache_transport": "request-scoped-memory-sliced-to-valid-prefix",
            "cache_bytes": cache_bytes,
            "session_load_seconds": 0.0,
            "persistent_session_load_seconds": self.worker.startup_seconds,
            "runtime_reused": True,
            "input_transport": "process-resident-memory",
            "decode_loop_seconds": float(native["decode_seconds"]),
            "llama_kv_import_seconds": float(native["kv_import_seconds"]),
            "llama_tokens_per_second": float(native["decode_tokens_per_second"]),
            "tokens": tokens,
            "text": text,
            "total_seconds": cache_seconds + float(native["decode_seconds"]),
        }
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
        return report

    def generate_arrays(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        valid_sequence: int,
        prefill_state: ResidentPrefillState,
        max_new_tokens: int,
        output_json: Path,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Continue one request using only its live prefill KV prefix."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        position_ids = np.asarray(position_ids, dtype=np.int64)
        model_sequence = self._validate_prompt(
            input_ids, attention_mask, position_ids, int(valid_sequence)
        )
        cache = prefill_state.cache
        if cache.length != valid_sequence:
            raise ValueError("resident KV cache and prompt valid sequence lengths differ")
        if cache.capacity < valid_sequence + max_new_tokens:
            raise ValueError("resident KV cache capacity is too small for decode")
        if (
            cache.num_layers != self.config.num_hidden_layers
            or cache.num_kv_heads != self.config.num_key_value_heads
            or cache.head_dim != self.config.head_dim
        ):
            raise ValueError("resident KV cache geometry does not match llama.cpp decoder")
        logits = np.asarray(prefill_state.initial_logits, dtype=np.float32)
        if logits.size != self.config.vocab_size:
            raise ValueError("prefill seed logits have incorrect vocabulary size")

        stream = TokenDeltaDecoder(self.tokenizer) if on_token is not None else None

        def token_callback(token: int) -> None:
            if stream is None:
                return
            delta = stream.append(token)
            if delta:
                assert on_token is not None
                on_token(delta)

        started = time.perf_counter()
        output_json = Path(output_json)
        with tempfile.TemporaryDirectory(
            prefix="llmdecode-kv-", dir=str(output_json.parent)
        ) as temporary:
            root = Path(temporary)
            logits_file = root / "last_token_logits.npy"
            positions_file = root / "position_ids_3d.npy"
            np.save(logits_file, np.ascontiguousarray(logits.reshape(-1), dtype=np.float32))
            # The worker supports the regular prompt-file ABI [3,1,Smax].
            np.save(positions_file, np.ascontiguousarray(position_ids[:, None, :], dtype=np.int64))
            keys: list[Path] = []
            values: list[Path] = []
            for layer, (key_store, value_store) in enumerate(zip(cache.keys, cache.values)):
                # Storage is [B, capacity, Hkv, D].  First slicing to
                # ``valid_sequence`` and then copying makes this a compact
                # [B,Hkv,S_valid,D] array; no padding is readable by native.
                key = np.ascontiguousarray(
                    key_store[:, :valid_sequence].transpose(0, 2, 1, 3), dtype=np.float16
                )
                value = np.ascontiguousarray(
                    value_store[:, :valid_sequence].transpose(0, 2, 1, 3), dtype=np.float16
                )
                expected = (1, cache.num_kv_heads, valid_sequence, cache.head_dim)
                if key.shape != expected or value.shape != expected:
                    raise AssertionError("valid-only KV conversion returned an invalid shape")
                key_file = root / f"layer_{layer:02d}.key.npy"
                value_file = root / f"layer_{layer:02d}.value.npy"
                np.save(key_file, key)
                np.save(value_file, value)
                keys.append(key_file)
                values.append(value_file)
            cache_seconds = time.perf_counter() - started
            native = self.worker.generate(
                logits=logits_file,
                positions=positions_file,
                keys=keys,
                values=values,
                kv_heads=cache.num_kv_heads,
                head_dim=cache.head_dim,
                prompt_tokens=valid_sequence,
                first_decode_position=int(position_ids[0, valid_sequence - 1]) + 1,
                max_new_tokens=max_new_tokens,
                descriptor=root / "request.desc",
                on_token_id=token_callback if stream is not None else None,
            )
        if stream is not None:
            tail = stream.finish()
            if tail:
                assert on_token is not None
                on_token(tail)
            text = stream.text
        else:
            text = self.tokenizer.decode(
                [int(value) for value in native["generated_token_ids"]],
                skip_special_tokens=True,
            )
        return self._report(
            native=native,
            model_sequence=model_sequence,
            cache_seconds=cache_seconds,
            cache_bytes=2 * cache.num_layers * cache.num_kv_heads * valid_sequence * cache.head_dim * 2,
            text=text,
            output_json=output_json,
        )

    def generate(
        self,
        *,
        prompt_dir: Path,
        prefill_dir: Path,
        max_new_tokens: int,
        output_json: Path,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Compatibility path for the legacy video runner's on-disk bundle."""
        prompt_dir = Path(prompt_dir)
        prefill_dir = Path(prefill_dir)
        metadata = PersistentGelabTFDLRuntime._read_manifest(
            prompt_dir / "metadata.json"
        )
        manifest = PersistentGelabTFDLRuntime._read_manifest(
            prefill_dir / "manifest.json"
        )
        valid_sequence = int(metadata["valid_seq_len"])
        input_ids = np.asarray(np.load(prompt_dir / "input_ids.npy"), dtype=np.int64)
        attention_mask = np.asarray(
            np.load(prompt_dir / "attention_mask.npy"), dtype=np.int64
        )
        position_ids = np.asarray(
            np.load(prompt_dir / "position_ids.npy"), dtype=np.int64
        )
        model_sequence = self._validate_prompt(
            input_ids, attention_mask, position_ids, valid_sequence
        )
        if int(manifest.get("valid_seq_len", -1)) != valid_sequence:
            raise ValueError("prefill and prompt valid sequence lengths differ")
        layers = [int(value) for value in manifest.get("layers", [])]
        if layers != list(range(self.config.num_hidden_layers)):
            raise ValueError("prefill KV layer order does not match llama.cpp decoder")
        cache_files = manifest.get("cache_files", {})
        keys: list[Path] = []
        values: list[Path] = []
        expected = (
            1,
            self.config.num_key_value_heads,
            valid_sequence,
            self.config.head_dim,
        )
        for layer in layers:
            entry = cache_files.get(str(layer), {})
            key = prefill_dir / str(entry.get("key", ""))
            value = prefill_dir / str(entry.get("value", ""))
            for kind, path in (("key", key), ("value", value)):
                tensor = np.load(path, mmap_mode="r")
                if tensor.dtype != np.float16 or tensor.shape != expected:
                    raise ValueError(
                        f"layer {layer} {kind}: expected FP16 {expected}, got "
                        f"{tensor.dtype} {tensor.shape}"
                    )
            keys.append(key)
            values.append(value)
        logits = prefill_dir / str(manifest.get("last_token_logits", ""))
        positions = prompt_dir / str(metadata.get("files", {}).get("position_ids_3d", ""))
        if not logits.is_file() or not positions.is_file():
            raise FileNotFoundError("prefill logits or MRoPE position file is missing")
        stream = TokenDeltaDecoder(self.tokenizer) if on_token is not None else None

        def token_callback(token: int) -> None:
            if stream is None:
                return
            delta = stream.append(token)
            if delta:
                assert on_token is not None
                on_token(delta)

        output_json = Path(output_json)
        with tempfile.TemporaryDirectory(
            prefix="llmdecode-kv-", dir=str(output_json.parent)
        ) as temporary:
            native = self.worker.generate(
                logits=logits,
                positions=positions,
                keys=keys,
                values=values,
                kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                prompt_tokens=valid_sequence,
                first_decode_position=int(position_ids[0, valid_sequence - 1]) + 1,
                max_new_tokens=max_new_tokens,
                descriptor=Path(temporary) / "request.desc",
                on_token_id=token_callback if stream is not None else None,
            )
        if stream is not None:
            tail = stream.finish()
            if tail:
                assert on_token is not None
                on_token(tail)
            text = stream.text
        else:
            text = self.tokenizer.decode(
                [int(value) for value in native["generated_token_ids"]],
                skip_special_tokens=True,
            )
        report = self._report(
            native=native,
            model_sequence=model_sequence,
            cache_seconds=0.0,
            cache_bytes=2
            * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * valid_sequence
            * self.config.head_dim
            * 2,
            text=text,
            output_json=output_json,
        )
        report["prompt_dir"] = str(prompt_dir.resolve())
        report["prefill_dir"] = str(prefill_dir.resolve())
        output_json.write_text(json.dumps(report, indent=2))
        return report


class PersistentGelabTFDLRuntime:
    """Keep one vision and every prefill TFDL executor resident in one process."""

    def __init__(
        self,
        *,
        model_path: Path,
        vision_fb_dir: Path,
        prefill_fb_dir: Path,
        addon_path: Path,
        vision_executor_config: dict[str, object],
        prefill_executor_config: dict[str, object],
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.vision_fb_dir = Path(vision_fb_dir).resolve()
        self.prefill_fb_dir = Path(prefill_fb_dir).resolve()
        self.vision_executor_config = deepcopy(vision_executor_config)
        self.prefill_executor_config = deepcopy(prefill_executor_config)
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"checkpoint is unavailable: {self.model_path}")
        if not Path(addon_path).is_file():
            raise FileNotFoundError(f"TFDL custom-op library is unavailable: {addon_path}")

        from TFDL2 import TFContext, TFExecutor
        from TFDL2.utils import LoadCustomOp

        LoadCustomOp(str(Path(addon_path).resolve()))
        self._TFContext = TFContext
        self._TFExecutor = TFExecutor
        self.contract = ModelContract.from_model(self.model_path)
        self.vision_manifest = self._read_manifest(
            self.vision_fb_dir / "manifest.single.json"
        )
        self.prefill_manifest = self._read_manifest(
            self.prefill_fb_dir / "manifest.json"
        )
        self.seq_len = int(self.prefill_manifest["seq_len"])
        self.prefill_config = prefill.QwenPrefillConfig.from_model(self.model_path)
        layers = [int(layer) for layer in self.prefill_manifest.get("layers", [])]
        expected_layers = list(range(self.prefill_config.num_hidden_layers))
        if layers != expected_layers:
            raise ValueError(
                f"prefill manifest layers={layers}, expected {expected_layers}"
            )
        if str(self.prefill_manifest.get("residual_layout", "BSC")).upper() != "BSC":
            raise ValueError("persistent GELab runtime currently requires BSC prefill FBs")

        started = time.perf_counter()
        vision_started = time.perf_counter()
        self.vision = SingleVisionExecutor(
            _topology_file(self.vision_fb_dir, self.vision_manifest),
            executor_config=self.vision_executor_config,
        )
        self.vision_compile_seconds = time.perf_counter() - vision_started

        self.prefill_contexts: list[Any] = []
        self.prefill_executors: list[Any] = []
        self.prefill_compile_seconds: list[float] = []
        for layer in expected_layers:
            artifact = self.prefill_fb_dir / f"layer_{layer:02d}_seq_{self.seq_len}.fb"
            if not artifact.is_file():
                raise FileNotFoundError(f"missing prefill FB: {artifact}")
            layer_started = time.perf_counter()
            context = self._TFContext(path=str(artifact))
            executor = self._TFExecutor(context, self.prefill_executor_config)
            self.prefill_compile_seconds.append(time.perf_counter() - layer_started)
            # Keep both context and executor alive.  TFDL parameters belong to
            # the context and executor reuse is the entire point of this path.
            self.prefill_contexts.append(context)
            self.prefill_executors.append(executor)
        self.startup = {
            "initialized": True,
            "seconds": time.perf_counter() - started,
            "vision_contexts": 1,
            "vision_executors": 1,
            "prefill_contexts": len(self.prefill_contexts),
            "prefill_executors": len(self.prefill_executors),
            "vision_compile_seconds": self.vision_compile_seconds,
            "prefill_compile_seconds": float(sum(self.prefill_compile_seconds)),
            "prefill_layer_compile_seconds": self.prefill_compile_seconds,
            "vision_executor_config": self.vision_executor_config,
            "prefill_executor_config": self.prefill_executor_config,
        }

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest: {path}")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"manifest must be a JSON object: {path}")
        return value

    def run_vision_arrays(
        self, pixels: np.ndarray, grid: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Run the resident visual FB directly from processor memory."""
        grid = np.asarray(grid, dtype=np.int64)
        if grid.shape != (1, 3):
            raise ValueError(f"vision grid must have shape [1,3], got {grid.shape}")
        temporal, grid_h, grid_w = (int(value) for value in grid[0])
        expected_grid = (
            int(self.vision_manifest["grid_h"]),
            int(self.vision_manifest["grid_w"]),
        )
        if (grid_h, grid_w) != expected_grid:
            raise ValueError(
                f"processor grid {(grid_h, grid_w)} does not match vision FB {expected_grid}"
            )
        sequence = grid_h * grid_w
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        expected_values = temporal * sequence * self.contract.patch_vector_size
        if pixels.size != expected_values:
            raise ValueError(
                f"pixel tensor has {pixels.size} values, expected {expected_values}"
            )
        pixels = pixels.reshape(temporal, sequence, self.contract.patch_vector_size)
        sin, cos = vision_rope(
            grid_h,
            grid_w,
            self.contract.vision_head_dim,
            self.contract.spatial_merge_size,
        )
        started = time.perf_counter()
        frames = [
            self.vision(pixels[index : index + 1], sin, cos)
            for index in range(temporal)
        ]
        execute_seconds = time.perf_counter() - started
        names = ("main", "deepstack_0", "deepstack_1", "deepstack_2")
        values = {
            name: np.concatenate([frame[index] for frame in frames], axis=1).reshape(
                -1, self.contract.hidden_size
            )
            for index, name in enumerate(names)
        }
        report = {
            "executor_lifetime": "process-resident",
            "executor_reused": True,
            "input_transport": "process-resident-memory",
            "grid_thw": grid.tolist(),
            "frames": temporal,
            "execute_seconds": execute_seconds,
            "shapes": {name: list(value.shape) for name, value in values.items()},
        }
        return values, report

    def run_vision(self, processor_bundle: Path, output_npz: Path) -> dict[str, Any]:
        """Run the resident single-FB vision encoder for every input frame."""
        bundle = Path(processor_bundle)
        media = self._read_manifest(bundle / "manifest.json")
        grid = np.load(bundle / media["files"]["grid_thw"]).astype(np.int64)
        temporal, grid_h, grid_w = (int(value) for value in grid[0])
        expected_grid = (
            int(self.vision_manifest["grid_h"]),
            int(self.vision_manifest["grid_w"]),
        )
        if (grid_h, grid_w) != expected_grid:
            raise ValueError(
                f"processor grid {(grid_h, grid_w)} does not match vision FB {expected_grid}"
            )
        sequence = grid_h * grid_w
        pixels = np.load(bundle / media["files"]["pixels"]).astype(np.float32)
        expected_values = temporal * sequence * self.contract.patch_vector_size
        if pixels.size != expected_values:
            raise ValueError(
                f"pixel tensor has {pixels.size} values, expected {expected_values}"
            )
        pixels = pixels.reshape(temporal, sequence, self.contract.patch_vector_size)
        sin, cos = vision_rope(
            grid_h,
            grid_w,
            self.contract.vision_head_dim,
            self.contract.spatial_merge_size,
        )
        started = time.perf_counter()
        frames = [self.vision(pixels[index : index + 1], sin, cos) for index in range(temporal)]
        execute_seconds = time.perf_counter() - started
        names = ("main", "deepstack_0", "deepstack_1", "deepstack_2")
        values = {
            name: np.concatenate([frame[index] for frame in frames], axis=1).reshape(
                -1, self.contract.hidden_size
            )
            for index, name in enumerate(names)
        }
        output_npz = Path(output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_npz, **values)
        report = {
            "executor_lifetime": "process-resident",
            "executor_reused": True,
            "grid_thw": grid.tolist(),
            "frames": temporal,
            "execute_seconds": execute_seconds,
            "output_npz": str(output_npz),
            "shapes": {name: list(value.shape) for name, value in values.items()},
        }
        output_npz.with_suffix(output_npz.suffix + ".json").write_text(
            json.dumps(report, indent=2)
        )
        return report

    def run_prefill(
        self,
        prompt_dir: Path,
        output_dir: Path,
        *,
        logits_runner: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Run all 36 resident prefill layers and write decoder-compatible KV."""
        prompt_dir = Path(prompt_dir)
        metadata = self._read_manifest(prompt_dir / "metadata.json")
        if metadata.get("format") != "qwen3-vl-prefill-prompt-v1":
            raise ValueError("unsupported prompt format")
        hidden = np.ascontiguousarray(np.load(prompt_dir / "hidden.npy"), dtype=np.float32)
        if hidden.shape != (1, self.seq_len, self.prefill_config.hidden_size):
            raise ValueError(
                f"prompt hidden shape {hidden.shape} does not match "
                f"[1,{self.seq_len},{self.prefill_config.hidden_size}]"
            )
        valid_sequence = int(metadata["valid_seq_len"])
        if not 0 < valid_sequence <= self.seq_len:
            raise ValueError("invalid prompt valid_seq_len")
        sin = np.ascontiguousarray(np.load(prompt_dir / "rope_sin.npy"), dtype=np.float32)
        cos = np.ascontiguousarray(np.load(prompt_dir / "rope_cos.npy"), dtype=np.float32)
        visual_mask = np.load(prompt_dir / "visual_mask.npy").astype(bool)
        deepstack = np.load(prompt_dir / "deepstack.npz")["features"].astype(np.float32)
        if deepstack.shape[0] != 3:
            raise ValueError(f"expected three DeepStack tensors, got {deepstack.shape}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_files: dict[str, dict[str, object]] = {}
        reports: list[dict[str, object]] = []
        started = time.perf_counter()
        for layer, executor in enumerate(self.prefill_executors):
            layer_started = time.perf_counter()
            hidden_next, key, value = prefill.execute_layer(executor, hidden, sin, cos)
            if layer < 3:
                hidden_next = inject_deepstack(hidden_next, visual_mask, deepstack[layer])
            hidden = np.ascontiguousarray(hidden_next, dtype=np.float32)
            key_valid = np.ascontiguousarray(
                key[:, :, :valid_sequence], dtype=np.float16
            )
            value_valid = np.ascontiguousarray(
                value[:, :, :valid_sequence], dtype=np.float16
            )
            key_name = f"layer_{layer:02d}.key.npy"
            value_name = f"layer_{layer:02d}.value.npy"
            np.save(output_dir / key_name, key_valid)
            np.save(output_dir / value_name, value_valid)
            cache_files[str(layer)] = {
                "key": key_name,
                "value": value_name,
                "shape": list(key_valid.shape),
                "dtype": "float16",
            }
            reports.append(
                {
                    "layer": layer,
                    "execute_seconds": time.perf_counter() - layer_started,
                    "deepstack_injected": layer < 3,
                    "executor_reused": True,
                }
            )

        final_hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        np.save(output_dir / "final_hidden.npy", final_hidden)
        last_hidden = np.ascontiguousarray(
            final_hidden[:, valid_sequence - 1 : valid_sequence], dtype=np.float32
        )
        logits_started = time.perf_counter()
        if logits_runner is None:
            raise ValueError(
                "persistent API prefill requires a resident final-head callback"
            )
        logits = logits_runner(last_hidden)
        np.save(output_dir / "last_token_logits.npy", np.asarray(logits, dtype=np.float32))
        manifest = {
            "format": PREFILL_FORMAT,
            "model_path": str(self.model_path),
            "prompt_dir": str(prompt_dir.resolve()),
            "fb_dir": str(self.prefill_fb_dir),
            "seq_len": valid_sequence,
            "valid_seq_len": valid_sequence,
            "model_seq_len": self.seq_len,
            "layers": list(range(self.prefill_config.num_hidden_layers)),
            "hidden_size": self.prefill_config.hidden_size,
            "num_key_value_heads": self.prefill_config.num_key_value_heads,
            "head_dim": self.prefill_config.head_dim,
            "residual_layout": "BSC",
            "deepstack_aware": True,
            "deepstack_injection_layers": [0, 1, 2],
            "cache_files": cache_files,
            "final_hidden": "final_hidden.npy",
            "last_token_logits": "last_token_logits.npy",
            "hardware": bool(self.prefill_executor_config.get("UseHardware", False)),
            "executor_config": self.prefill_executor_config,
            "executor_lifetime": "process-resident",
            "runtime_reused": True,
            "logits_engine": "runtime-assets-rmsnorm-tied-bf16-head",
            "logits_seconds": time.perf_counter() - logits_started,
            "total_seconds": time.perf_counter() - started,
            "layer_reports": reports,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def run_prefill_arrays(
        self,
        prompt: dict[str, Any],
        *,
        max_new_tokens: int,
        logits_runner: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> ResidentPrefillState:
        """Run all resident layers from in-memory prompt tensors.

        The returned KV state is owned by the caller's one decode request; no
        K/V, final hidden state, or first-token logits are serialized.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        hidden = np.ascontiguousarray(prompt["hidden"], dtype=np.float32)
        if hidden.shape != (1, self.seq_len, self.prefill_config.hidden_size):
            raise ValueError(
                f"resident prompt hidden shape {hidden.shape} does not match "
                f"[1,{self.seq_len},{self.prefill_config.hidden_size}]"
            )
        valid_sequence = int(prompt["valid_seq_len"])
        if not 0 < valid_sequence <= self.seq_len:
            raise ValueError("invalid resident prompt valid_seq_len")
        sin = np.ascontiguousarray(prompt["rope_sin"], dtype=np.float32)
        cos = np.ascontiguousarray(prompt["rope_cos"], dtype=np.float32)
        visual_mask = np.asarray(prompt["visual_mask"], dtype=bool)
        deepstack = np.asarray(prompt["deepstack"], dtype=np.float32)
        if deepstack.shape[0] != 3:
            raise ValueError(f"expected three DeepStack tensors, got {deepstack.shape}")

        cache = RequestKvCache(
            num_layers=self.prefill_config.num_hidden_layers,
            num_kv_heads=self.prefill_config.num_key_value_heads,
            head_dim=self.prefill_config.head_dim,
            prompt_length=valid_sequence,
            capacity=valid_sequence + max_new_tokens,
        )
        reports: list[dict[str, object]] = []
        started = time.perf_counter()
        for layer, executor in enumerate(self.prefill_executors):
            layer_started = time.perf_counter()
            hidden_next, key, value = prefill.execute_layer(executor, hidden, sin, cos)
            if layer < 3:
                hidden_next = inject_deepstack(hidden_next, visual_mask, deepstack[layer])
            hidden = np.ascontiguousarray(hidden_next, dtype=np.float32)
            cache.set_prefill(
                layer,
                np.ascontiguousarray(key[:, :, :valid_sequence], dtype=np.float16),
                np.ascontiguousarray(
                    value[:, :, :valid_sequence], dtype=np.float16
                ),
            )
            reports.append(
                {
                    "layer": layer,
                    "execute_seconds": time.perf_counter() - layer_started,
                    "deepstack_injected": layer < 3,
                    "executor_reused": True,
                }
            )
        last_hidden = np.ascontiguousarray(
            hidden[:, valid_sequence - 1 : valid_sequence], dtype=np.float32
        )
        if logits_runner is None:
            raise ValueError(
                "persistent API prefill requires a resident final-head callback"
            )
        logits_started = time.perf_counter()
        logits = np.ascontiguousarray(logits_runner(last_hidden), dtype=np.float32)
        manifest = {
            "format": PREFILL_FORMAT,
            "model_path": str(self.model_path),
            "prompt_dir": "in-memory",
            "cache_transport": "request-scoped-memory",
            "cache_bytes": cache.nbytes,
            "fb_dir": str(self.prefill_fb_dir),
            "seq_len": valid_sequence,
            "valid_seq_len": valid_sequence,
            "model_seq_len": self.seq_len,
            "layers": list(range(self.prefill_config.num_hidden_layers)),
            "hidden_size": self.prefill_config.hidden_size,
            "num_key_value_heads": self.prefill_config.num_key_value_heads,
            "head_dim": self.prefill_config.head_dim,
            "residual_layout": "BSC",
            "deepstack_aware": True,
            "deepstack_injection_layers": [0, 1, 2],
            "hardware": bool(self.prefill_executor_config.get("UseHardware", False)),
            "executor_config": self.prefill_executor_config,
            "executor_lifetime": "process-resident",
            "runtime_reused": True,
            "input_transport": "process-resident-memory",
            "logits_engine": "runtime-assets-rmsnorm-tied-bf16-head",
            "logits_seconds": time.perf_counter() - logits_started,
            "total_seconds": time.perf_counter() - started,
            "layer_reports": reports,
        }
        return ResidentPrefillState(
            manifest=manifest,
            cache=cache,
            initial_logits=logits,
        )
