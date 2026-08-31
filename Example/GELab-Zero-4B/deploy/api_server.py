#!/usr/bin/env python3
"""Small local HTTP API for the process-resident GELab TFDL runtime.

The service intentionally permits only one generation at a time.  It builds
the single-FB vision executor and all 36 prefill executors at startup, then
reuses them for every request.  This keeps NPU ownership deterministic and
removes graph construction from first-request latency.

POST /v1/generate accepts JSON with image_path, or a trusted
visual_features_npz data URL for visual-encoder A/B debugging, plus optional
question, system, and max_new_tokens. POST
/v1/chat/completions`` exposes the same pipeline with the OpenAI Chat
Completions schema.  Paths are local to the machine hosting the service;
upload handling is deliberately left to a front-end/reverse proxy.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse, urlsplit
from urllib.request import Request, urlopen

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
DEFAULT_OPENAI_MODEL_ID = "gelab-zero-4b"


def _openai_content(content: Any) -> tuple[list[str], list[str]]:
    """Read the text and local image paths from one OpenAI content value."""
    if isinstance(content, str):
        return [content], []
    if not isinstance(content, list):
        raise ValueError("each message content must be a string or a content-part list")

    texts: list[str] = []
    image_paths: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("each message content part must be an object")
        part_type = part.get("type")
        if part_type in ("text", "input_text"):
            value = part.get("text")
            if value is None and part_type == "input_text":
                value = part.get("input_text")
            if not isinstance(value, str):
                raise ValueError(f"{part_type} content requires a string text field")
            texts.append(value)
            continue
        if part_type != "image_url":
            raise ValueError(
                "only text/input_text and image_url content parts are supported"
            )
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url:
            raise ValueError("image_url content requires image_url.url")
        parsed = urlparse(url)
        if parsed.scheme == "data":
            image_paths.append(url)
        elif parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("file image_url must not specify a remote host")
            image_paths.append(unquote(parsed.path))
        elif parsed.scheme in ("http", "https"):
            image_paths.append(url)
        elif parsed.scheme:
            raise ValueError(
                "image_url must be an absolute path, data:, file:///, http://, or https:// URL"
            )
        else:
            image_paths.append(url)
    return texts, image_paths


def parse_openai_chat_request(
    payload: dict[str, Any], default_model: str, max_new_tokens_limit: int = 128
) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions request to :meth:`generate` input.

    The model has one-image visual input and does not retain a conversational
    KV cache between requests.  We therefore turn preceding text turns into a
    compact textual transcript, while allowing exactly one local image.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    model = payload.get("model", default_model)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")

    system_parts: list[str] = []
    transcript: list[str] = []
    image_paths: list[str] = []
    supported_roles = {"system", "developer", "user", "assistant"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in supported_roles:
            raise ValueError(
                f"messages[{index}].role must be one of {sorted(supported_roles)}"
            )
        texts, paths = _openai_content(message.get("content"))
        image_paths.extend(paths)
        text = "\n".join(item for item in texts if item)
        if role in ("system", "developer"):
            if text:
                system_parts.append(text)
        elif text:
            transcript.append(f"{role.title()}: {text}")

    if len(image_paths) != 1:
        raise ValueError(
            "this GELab deployment supports exactly one image_url across messages"
        )
    if not transcript:
        raise ValueError("at least one user or assistant text content part is required")

    max_tokens = payload.get(
        "max_completion_tokens", payload.get("max_tokens", payload.get("max_new_tokens", 1))
    )
    try:
        max_new_tokens = int(max_tokens)
    except (TypeError, ValueError) as error:
        raise ValueError("max_completion_tokens/max_tokens must be an integer") from error
    if not 1 <= max_new_tokens <= max_new_tokens_limit:
        raise ValueError(
            "max_completion_tokens/max_tokens must be in "
            f"[1, {max_new_tokens_limit}]"
        )

    return {
        "model": model,
        "stream": stream,
        "image_path": image_paths[0],
        "question": "\n".join(transcript),
        "system": "\n".join(system_parts) or "You are a helpful GUI agent.",
        "max_new_tokens": max_new_tokens,
    }


def openai_completion_response(
    request: dict[str, Any], pipeline: dict[str, Any], completion_id: str, created: int
) -> dict[str, Any]:
    """Make a standard non-stream OpenAI completion response."""
    result = pipeline["result"]
    completion_tokens = int(result["generated_tokens"])
    prompt_tokens = int(result["prompt_tokens"])
    finish_reason = (
        "length" if completion_tokens >= int(request["max_new_tokens"]) else "stop"
    )
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def openai_completion_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None,
) -> dict[str, Any]:
    """Make one standard Chat Completions SSE payload."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-path",
        help=(
            "deprecated compatibility alias for an exported runtime-assets directory; "
            "the API never consumes a safetensors checkpoint"
        ),
    )
    parser.add_argument("--config", default=str(DEPLOY_DIR / "deployment.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--software", action="store_true", help="override executor UseHardware=false")
    mode.add_argument("--hardware", action="store_true", help="override executor UseHardware=true")
    parser.add_argument("--cpu-limit", type=int, help="override cpuLimit in both executor configs")
    parser.add_argument(
        "--prefill-chunk-layers",
        type=int,
        help="legacy runner setting; ignored by the process-resident API",
    )
    parser.add_argument(
        "--lazy-executors",
        action="store_true",
        help="defer TFDL/ORT runtime construction until the first request (diagnostic only)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--jobs-dir", default=str(DEPLOY_DIR / "var/jobs"))
    return parser.parse_args()


class GelabService:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.port <= 0 or args.timeout_seconds <= 0:
            raise ValueError("--port/--timeout-seconds must be positive")
        self.config_path = Path(args.config).resolve()
        config = json.loads(self.config_path.read_text())
        self.config = config
        self.root = self.config_path.parent
        api = config.get("api", {})
        if not isinstance(api, dict):
            raise ValueError("api config must be a JSON object when provided")
        self.openai_model_id = str(api.get("model_id", DEFAULT_OPENAI_MODEL_ID))
        if not self.openai_model_id:
            raise ValueError("api.model_id must be a non-empty string")
        remote_media = api.get("remote_media", {})
        if not isinstance(remote_media, dict):
            raise ValueError("api.remote_media must be a JSON object when provided")
        self.remote_media_enabled = bool(remote_media.get("enabled", True))
        self.remote_media_timeout_seconds = int(remote_media.get("timeout_seconds", 30))
        self.remote_media_max_bytes = int(remote_media.get("max_bytes", 32 * 1024 * 1024))
        self.max_request_bytes = int(api.get("max_request_bytes", 32 * 1024 * 1024))
        if (
            self.remote_media_timeout_seconds <= 0
            or self.remote_media_max_bytes <= 0
            or self.max_request_bytes <= 0
        ):
            raise ValueError(
                "api remote-media timeout/max bytes and max_request_bytes must be positive"
            )
        tfdl = config["tfdl"]
        vision = tfdl["vision"]
        prefill = tfdl["prefill"]
        decode = config["decode"]
        self.max_new_tokens = int(decode.get("max_new_tokens", 128))
        if self.max_new_tokens <= 0:
            raise ValueError("decode.max_new_tokens must be positive")
        self.runtime_assets_dir = self.root / str(
            config.get("runtime_assets_dir", "model/runtime")
        )
        requested_assets = (
            args.model_path
            or os.environ.get("GELAB_RUNTIME_ASSETS")
            or os.environ.get("GELAB_MODEL_PATH")
        )
        self.model_path = (
            Path(requested_assets).expanduser().resolve()
            if requested_assets
            else self.runtime_assets_dir.resolve()
        )
        if not (self.model_path / "manifest.json").is_file():
            raise FileNotFoundError(
                "API requires exported runtime assets, not a safetensors checkpoint: "
                f"{self.model_path}. Run export_models.py --model-path MODEL_DIR "
                "--stage runtime, or pass that runtime-assets directory."
            )
        self.addon_path = self.root / str(tfdl["addon"])
        self.vision_fb_dir = self.root / str(vision["fb_dir"])
        self.prefill_fb_dir = self.root / str(prefill["fb_dir"])
        self.decoder_dir = self.root / str(config["decode"]["decoder_dir"])
        self.vision_executor = deepcopy(vision.get("executor"))
        self.prefill_executor = deepcopy(prefill.get("executor"))
        if not isinstance(self.vision_executor, dict) or not isinstance(self.prefill_executor, dict):
            raise ValueError("tfdl vision/prefill executor configs must be JSON objects")
        self.use_hardware_override = (
            False if args.software else True if args.hardware else None
        )
        self.cpu_limit_override = args.cpu_limit
        if self.use_hardware_override is not None:
            self.vision_executor["UseHardware"] = self.use_hardware_override
            self.prefill_executor["UseHardware"] = self.use_hardware_override
        if self.cpu_limit_override is not None:
            if self.cpu_limit_override <= 0:
                raise ValueError("--cpu-limit must be positive")
            self.vision_executor["cpuLimit"] = self.cpu_limit_override
            self.prefill_executor["cpuLimit"] = self.cpu_limit_override
        for stage, executor in (
            ("vision", self.vision_executor),
            ("prefill", self.prefill_executor),
        ):
            if int(executor.get("cpuLimit", 0)) <= 0:
                raise ValueError(f"tfdl.{stage}.executor.cpuLimit must be positive")
        self.configured_prefill_chunk_layers = int(
            args.prefill_chunk_layers
            if args.prefill_chunk_layers is not None
            else prefill["prefill_chunk_layers"]
        )
        if self.configured_prefill_chunk_layers < 0:
            raise ValueError("chunk layers must be non-negative")
        self.timeout_seconds = int(args.timeout_seconds)
        self.jobs_dir = Path(args.jobs_dir).resolve()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.runtime = None
        self.decoder_runtime = None
        self.input_builder = None
        self.runtime_startup: dict[str, Any] = {
            "initialized": False,
            "lazy": bool(args.lazy_executors),
        }
        if not args.lazy_executors:
            self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        """Construct TFDL executors and the selected persistent decoder once."""
        if (
            self.runtime is not None
            and self.decoder_runtime is not None
            and self.input_builder is not None
        ):
            return
        from persistent_tfdl_runtime import (
            PersistentGelabTFDLRuntime,
            PersistentLlamaCppDecoder,
            ResidentGelabInputBuilder,
        )

        decode = self.config["decode"]
        engine = str(decode.get("engine", "llama.cpp-external-kv"))
        if engine != "llama.cpp-external-kv":
            raise ValueError(f"unsupported decode.engine: {engine}")
        try:
            import tfdl_llmdecode  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "install the project-local deploy/wheels/tfdl_llmdecode-*.whl "
                "before starting the GELab API server"
            ) from error
        print("[startup] loading persistent llama.cpp decoder...", flush=True)
        self.decoder_runtime = PersistentLlamaCppDecoder(
            model_path=self.model_path,
            decoder_dir=self.decoder_dir,
            threads=int(decode["threads"]),
            kv_cache_type=str(decode.get("cache_dtype", "float16")).replace(
                "float16", "fp16"
            ),
        )
        print(
            "[startup] llama.cpp decoder ready in "
            f"{self.decoder_runtime.startup.get('worker_model_load_seconds', 0.0):.2f}s; "
            "building TFDL vision/prefill executors...",
            flush=True,
        )
        self.runtime = PersistentGelabTFDLRuntime(
            model_path=self.model_path,
            vision_fb_dir=self.vision_fb_dir,
            prefill_fb_dir=self.prefill_fb_dir,
            addon_path=self.addon_path,
            vision_executor_config=self.vision_executor,
            prefill_executor_config=self.prefill_executor,
        )
        self.input_builder = ResidentGelabInputBuilder(
            model_path=self.model_path,
            seq_len=int(self.config["tfdl"]["prefill"]["seq_len"]),
            tokenizer=self.decoder_runtime.tokenizer,
            embeddings=self.decoder_runtime.embeddings,
        )
        self.runtime_startup = dict(self.runtime.startup)
        self.runtime_startup["decode"] = dict(self.decoder_runtime.startup)
        self.runtime_startup["input_builder"] = dict(self.input_builder.startup)
        print("[startup] all GELab executors are ready.", flush=True)

    def health(self) -> dict[str, Any]:
        tfdl = self.config["tfdl"]
        return {
            "status": "ok",
            "vision_executor": self.vision_executor,
            "prefill_executor": self.prefill_executor,
            "tfdl_runtime": self.runtime_startup,
            "decode_runtime_ready": self.decoder_runtime is not None,
            "input_builder_ready": self.input_builder is not None,
            "prefill_execution": "all 36 layers in one resident API process",
            "legacy_prefill_chunk_layers": self.configured_prefill_chunk_layers,
            "model_path": str(self.model_path),
            "runtime_assets_dir": str(self.runtime_assets_dir),
            "runtime_assets_available": (self.runtime_assets_dir / "manifest.json").is_file(),
            "model_available": (self.model_path / "config.json").is_file(),
            "vision_available": (self.vision_fb_dir / "manifest.single.json").is_file(),
            "prefill_available": (self.prefill_fb_dir / "manifest.json").is_file(),
            "decode_available": (self.decoder_dir / "manifest.json").is_file(),
            "decode_engine": self.config["decode"].get("engine"),
            "gqa": tfdl["prefill"].get("gqa"),
            "openai_model_id": self.openai_model_id,
            "remote_media": {
                "enabled": self.remote_media_enabled,
                "timeout_seconds": self.remote_media_timeout_seconds,
                "max_bytes": self.remote_media_max_bytes,
            },
            "max_request_bytes": self.max_request_bytes,
            "max_new_tokens": self.max_new_tokens,
        }

    def openai_models(self) -> dict[str, Any]:
        """Return the OpenAI-compatible models-list response."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.openai_model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "gelab-tfdl",
                }
            ],
        }

    @staticmethod
    def _media_suffix(content_type: str, kind: str, source_url: str = "") -> str:
        suffix = Path(urlsplit(source_url).path).suffix.lower()
        if suffix and len(suffix) <= 10:
            return suffix
        return mimetypes.guess_extension(content_type) or (
            ".png" if kind == "image" else ".mp4"
        )

    def _validate_media_type(self, content_type: str, kind: str) -> None:
        expected_prefix = "image/" if kind == "image" else "video/"
        if (
            content_type
            and content_type != "application/octet-stream"
            and not content_type.startswith(expected_prefix)
        ):
            raise ValueError(
                f"remote {kind} Content-Type must begin with {expected_prefix}, got {content_type}"
            )

    def _materialize_media(
        self, media_reference: str, kind: str, job: Path
    ) -> tuple[Path, dict[str, Any] | None]:
        """Resolve a local path or materialize data/HTTP media inside one job."""
        parsed = urlparse(media_reference)
        if parsed.scheme not in ("data", "http", "https"):
            media = Path(media_reference).expanduser().resolve()
            if not media.is_file():
                raise FileNotFoundError(f"media file does not exist: {media}")
            return media, None

        target_dir = job / "input-media"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "input"
        if parsed.scheme == "data":
            try:
                header, encoded = media_reference.split(",", 1)
            except ValueError as error:
                raise ValueError("data image_url must contain a comma before base64 data") from error
            if not header.lower().endswith(";base64"):
                raise ValueError("data image_url must use base64 encoding")
            content_type = header[5:-7].lower()
            self._validate_media_type(content_type, kind)
            # Reject over-limit input before decoding the base64 payload into RAM.
            if len(encoded) > ((self.remote_media_max_bytes * 4 + 2) // 3) + 8:
                raise ValueError(
                    f"base64 media exceeds {self.remote_media_max_bytes} byte limit"
                )
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("data image_url contains invalid base64") from error
            if len(raw) > self.remote_media_max_bytes:
                raise ValueError(
                    f"base64 media exceeds {self.remote_media_max_bytes} byte limit"
                )
            materialized = target.with_suffix(self._media_suffix(content_type, kind))
            materialized.write_bytes(raw)
            return materialized, {
                "source": "data-url",
                "content_type": content_type,
                "bytes": len(raw),
                "path": str(materialized),
            }

        if not self.remote_media_enabled:
            raise ValueError("remote HTTP(S) media is disabled by api.remote_media.enabled")
        request = Request(
            media_reference,
            headers={"User-Agent": "GELab-TFDL/1.0 remote-media-loader"},
        )
        total = 0
        content_type = ""
        final_url = media_reference
        try:
            with urlopen(request, timeout=self.remote_media_timeout_seconds) as response:
                final_url = response.geturl()
                if urlparse(final_url).scheme not in ("http", "https"):
                    raise ValueError("remote media redirect must remain HTTP(S)")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                self._validate_media_type(content_type, kind)
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and int(declared_length) > self.remote_media_max_bytes:
                    raise ValueError(
                        f"remote media exceeds {self.remote_media_max_bytes} byte limit"
                    )
                with target.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > self.remote_media_max_bytes:
                            raise ValueError(
                                f"remote media exceeds {self.remote_media_max_bytes} byte limit"
                            )
                        output.write(chunk)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            target.unlink(missing_ok=True)
            raise ValueError(f"failed to download remote media {media_reference}: {error}") from error

        materialized = target.with_suffix(self._media_suffix(content_type, kind, final_url))
        target.rename(materialized)
        return materialized, {
            "source": "http-url",
            "url": media_reference,
            "final_url": final_url,
            "content_type": content_type or None,
            "bytes": total,
            "path": str(materialized),
        }

    def _decode_external_visual_features(
        self, value: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Read one trusted GPU-vision NPZ data URL directly into RAM."""
        if not isinstance(value, str) or not value:
            raise ValueError("visual_features_npz must be a non-empty data URL string")
        try:
            header, encoded = value.split(",", 1)
        except ValueError as error:
            raise ValueError(
                "visual_features_npz must be data:application/x-npz;base64,..."
            ) from error
        accepted_headers = {
            "data:application/x-npz;base64",
            "data:application/octet-stream;base64",
        }
        if header.lower() not in accepted_headers:
            raise ValueError(
                "visual_features_npz must use data:application/x-npz;base64 encoding"
            )
        if len(encoded) > ((self.max_request_bytes * 4 + 2) // 3) + 8:
            raise ValueError(f"visual feature payload exceeds {self.max_request_bytes} bytes")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("visual_features_npz contains invalid base64") from error
        if len(raw) > self.max_request_bytes:
            raise ValueError(f"visual feature payload exceeds {self.max_request_bytes} bytes")
        required = {"main", "deepstack_0", "deepstack_1", "deepstack_2", "grid_thw"}
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                names = set(archive.files)
                if names != required:
                    raise ValueError(
                        f"visual feature NPZ must contain exactly {sorted(required)}, "
                        f"got {sorted(names)}"
                    )
                grid = np.asarray(archive["grid_thw"], dtype=np.int64)
                values = {
                    name: np.ascontiguousarray(archive[name], dtype=np.float32)
                    for name in required
                    if name != "grid_thw"
                }
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"invalid visual feature NPZ: {error}") from error

        assert self.input_builder is not None
        contract = self.input_builder.contract
        expected_grid = np.asarray(
            [[
                1,
                int(self.config["tfdl"]["vision"]["height"]) // contract.vision_patch_size,
                int(self.config["tfdl"]["vision"]["width"]) // contract.vision_patch_size,
            ]],
            dtype=np.int64,
        )
        if grid.shape != (1, 3) or not np.array_equal(grid, expected_grid):
            raise ValueError(
                f"external feature grid {grid.tolist()} does not match deployment "
                f"grid {expected_grid.tolist()}"
            )
        expected_shape = (
            int(np.prod(grid) // contract.spatial_merge_size**2),
            contract.hidden_size,
        )
        invalid = {
            name: list(item.shape)
            for name, item in values.items()
            if item.shape != expected_shape or not np.isfinite(item).all()
        }
        if invalid:
            raise ValueError(
                f"external visual tensors must be finite {expected_shape}, got {invalid}"
            )
        return values, {
            "input_transport": "external-vision-features-npz",
            "bytes": len(raw),
            "grid_thw": grid.tolist(),
            "visual_tokens": expected_shape[0],
            "hidden_size": expected_shape[1],
            "dtype": "float32",
        }

    def _generate_image_resident(
        self,
        payload: dict[str, Any],
        *,
        on_decode_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run the image path without request-spawned Python subprocesses.

        The media processor, tokenizer, embedding table, TFDL executors,
        llama.cpp decoder, and request-scoped K/V tensors all live in this server
        process. Nothing from prefill is persisted between requests.
        """
        image_reference = str(payload["image_path"])
        question = str(
            payload.get("question")
            or "Describe the screen and identify the next useful action."
        )
        system = str(payload.get("system") or "You are a helpful GUI agent.")
        max_new_tokens = int(payload.get("max_new_tokens", 1))
        if not 1 <= max_new_tokens <= self.max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be in [1, {self.max_new_tokens}]"
            )
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"GELab runtime assets are unavailable: {self.model_path}")

        job = self.jobs_dir / (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        started = time.perf_counter()
        stages: dict[str, float] = {}
        logs: list[str] = []
        # TFDL vision/prefill executors share the NPU and must remain
        # serialized.  The request-specific llama.cpp decode runs after this
        # critical section and can overlap with the next request's NPU work.
        # Startup still happens before the listen socket unless
        # --lazy-executors was selected.
        with self.lock:
            self._initialize_runtime()
            assert self.runtime is not None
            assert self.decoder_runtime is not None
            assert self.input_builder is not None
            job.mkdir(parents=True, exist_ok=False)

            stage_started = time.perf_counter()
            image, materialization = self._materialize_media(
                image_reference, "image", job
            )
            stages["materialize_media"] = time.perf_counter() - stage_started
            if materialization is not None:
                logs.append(
                    "[media-materialization] " + json.dumps(materialization) + "\n"
                )

            vision_config = self.config["tfdl"]["vision"]
            stage_started = time.perf_counter()
            media, media_report = self.input_builder.prepare_image(
                image,
                question=question,
                system=system,
                height=int(vision_config["height"]),
                width=int(vision_config["width"]),
            )
            stages["prepare_media"] = time.perf_counter() - stage_started
            logs.append("[resident-media] " + json.dumps(media_report) + "\n")

            stage_started = time.perf_counter()
            visual, vision_report = self.runtime.run_vision_arrays(
                media["pixels"], media["grid_thw"]
            )
            stages["vision"] = time.perf_counter() - stage_started
            logs.append("[resident-vision] " + json.dumps(vision_report) + "\n")

            stage_started = time.perf_counter()
            prompt, prompt_report = self.input_builder.build_prefill(media, visual)
            stages["prepare_prompt"] = time.perf_counter() - stage_started
            logs.append("[resident-prompt] " + json.dumps(prompt_report) + "\n")

            stage_started = time.perf_counter()
            prefill_state = self.runtime.run_prefill_arrays(
                prompt,
                max_new_tokens=max_new_tokens,
                logits_runner=self.decoder_runtime.initial_logits,
            )
            stages["prefill"] = time.perf_counter() - stage_started
            prefill_manifest = prefill_state.manifest
            logs.append(
                "[resident-prefill] "
                + json.dumps(
                    {
                        "layers": len(prefill_manifest["layers"]),
                        "total_seconds": prefill_manifest["total_seconds"],
                        "logits_seconds": prefill_manifest["logits_seconds"],
                        "executor_lifetime": prefill_manifest["executor_lifetime"],
                        "input_transport": prefill_manifest["input_transport"],
                        "cache_transport": prefill_manifest["cache_transport"],
                        "cache_bytes": prefill_manifest["cache_bytes"],
                    }
                )
                + "\n"
            )

        stage_started = time.perf_counter()
        decode_report = self.decoder_runtime.generate_arrays(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            position_ids=prompt["position_ids"],
            valid_sequence=int(prompt["valid_seq_len"]),
            prefill_state=prefill_state,
            max_new_tokens=max_new_tokens,
            output_json=job / "result.json",
            on_token=on_decode_token,
        )
        stages["decode"] = time.perf_counter() - stage_started
        logs.append(
            "[resident-decode] "
            + json.dumps(
                {
                    "generated_tokens": decode_report["generated_tokens"],
                    "total_seconds": decode_report["total_seconds"],
                    "runtime_reused": decode_report["runtime_reused"],
                    "session_load_seconds": decode_report["session_load_seconds"],
                }
            )
            + "\n"
        )

        duration = time.perf_counter() - started
        result_path = job / "result.json"
        if not result_path.is_file():
            raise RuntimeError("pipeline completed without result.json\n" + "".join(logs)[-12000:])
        logs.append("[stage-timings] " + json.dumps(stages) + "\n")
        return {
            "job_dir": str(job),
            "duration_seconds": duration,
            "stage_seconds": stages,
            "result": json.loads(result_path.read_text()),
            "pipeline_log": "".join(logs),
        }

    def _generate_external_visual_features_resident(
        self,
        payload: dict[str, Any],
        *,
        on_decode_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run prefill/decode from a validated external GPU-vision NPZ bundle."""
        question = str(
            payload.get("question")
            or "Describe the screen and identify the next useful action."
        )
        system = str(payload.get("system") or "You are a helpful GUI agent.")
        max_new_tokens = int(payload.get("max_new_tokens", 1))
        if not 1 <= max_new_tokens <= self.max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be in [1, {self.max_new_tokens}]"
            )
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"GELab runtime assets are unavailable: {self.model_path}")

        job = self.jobs_dir / (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        started = time.perf_counter()
        stages: dict[str, float] = {}
        logs: list[str] = []
        with self.lock:
            self._initialize_runtime()
            assert self.runtime is not None
            assert self.decoder_runtime is not None
            assert self.input_builder is not None
            job.mkdir(parents=True, exist_ok=False)

            stage_started = time.perf_counter()
            visual, visual_report = self._decode_external_visual_features(
                payload["visual_features_npz"]
            )
            stages["decode_external_visual_features"] = time.perf_counter() - stage_started
            logs.append(
                "[external-vision-features] " + json.dumps(visual_report) + "\n"
            )

            vision_config = self.config["tfdl"]["vision"]
            stage_started = time.perf_counter()
            media, media_report = self.input_builder.prepare_external_visual_prompt(
                question=question,
                system=system,
                height=int(vision_config["height"]),
                width=int(vision_config["width"]),
                visual_tokens=int(visual_report["visual_tokens"]),
            )
            prompt, prompt_report = self.input_builder.build_prefill(media, visual)
            stages["prepare_prompt"] = time.perf_counter() - stage_started
            logs.append("[external-visual-prompt] " + json.dumps(media_report) + "\n")
            logs.append("[resident-prompt] " + json.dumps(prompt_report) + "\n")

            stage_started = time.perf_counter()
            prefill_state = self.runtime.run_prefill_arrays(
                prompt,
                max_new_tokens=max_new_tokens,
                logits_runner=self.decoder_runtime.initial_logits,
            )
            stages["prefill"] = time.perf_counter() - stage_started
            prefill_manifest = prefill_state.manifest
            logs.append(
                "[resident-prefill] "
                + json.dumps(
                    {
                        "layers": len(prefill_manifest["layers"]),
                        "total_seconds": prefill_manifest["total_seconds"],
                        "logits_seconds": prefill_manifest["logits_seconds"],
                        "executor_lifetime": prefill_manifest["executor_lifetime"],
                        "input_transport": prefill_manifest["input_transport"],
                        "cache_transport": prefill_manifest["cache_transport"],
                        "cache_bytes": prefill_manifest["cache_bytes"],
                    }
                )
                + "\n"
            )

        # External visual features skip the NPU vision stage, but their
        # prefill still uses the same NPU critical section.  Decode is again
        # intentionally outside it.
        stage_started = time.perf_counter()
        decode_report = self.decoder_runtime.generate_arrays(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            position_ids=prompt["position_ids"],
            valid_sequence=int(prompt["valid_seq_len"]),
            prefill_state=prefill_state,
            max_new_tokens=max_new_tokens,
            output_json=job / "result.json",
            on_token=on_decode_token,
        )
        stages["decode"] = time.perf_counter() - stage_started
        logs.append(
            "[resident-decode] "
            + json.dumps(
                {
                    "generated_tokens": decode_report["generated_tokens"],
                    "total_seconds": decode_report["total_seconds"],
                    "runtime_reused": decode_report["runtime_reused"],
                    "session_load_seconds": decode_report["session_load_seconds"],
                }
            )
            + "\n"
        )

        duration = time.perf_counter() - started
        result_path = job / "result.json"
        if not result_path.is_file():
            raise RuntimeError("pipeline completed without result.json\n" + "".join(logs)[-12000:])
        logs.append("[stage-timings] " + json.dumps(stages) + "\n")
        return {
            "job_dir": str(job),
            "duration_seconds": duration,
            "stage_seconds": stages,
            "result": json.loads(result_path.read_text()),
            "pipeline_log": "".join(logs),
        }

    def generate(
        self,
        payload: dict[str, Any],
        *,
        on_decode_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a fixed-size image request or trusted external features."""
        if "visual_features_npz" in payload:
            if payload.get("image_path") or payload.get("video_path"):
                raise ValueError(
                    "visual_features_npz cannot be combined with image_path"
                )
            return self._generate_external_visual_features_resident(
                payload, on_decode_token=on_decode_token
            )
        image = payload.get("image_path")
        if payload.get("video_path"):
            raise ValueError("the fixed GELab deployment supports image requests only")
        if not image:
            raise ValueError("provide image_path or visual_features_npz")
        return self._generate_image_resident(
            payload, on_decode_token=on_decode_token
        )

    def chat_completion_from_request(
        self,
        request: dict[str, Any],
        *,
        on_decode_token: Callable[[str], None] | None = None,
        completion_id: str | None = None,
        created: int | None = None,
    ) -> dict[str, Any]:
        """Run one already-validated Chat Completions request."""
        pipeline = self.generate(request, on_decode_token=on_decode_token)
        return openai_completion_response(
            request=request,
            pipeline=pipeline,
            completion_id=completion_id or "chatcmpl-" + uuid.uuid4().hex,
            created=created if created is not None else int(time.time()),
        )

    def chat_completion(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one non-streaming or buffered Chat Completions request."""
        request = parse_openai_chat_request(
            payload, self.openai_model_id, self.max_new_tokens
        )
        return self.chat_completion_from_request(request), bool(request["stream"])


def handler_factory(service: GelabService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GELabTFDL/1.0"

        def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _reply_openai_error(
            self, status: HTTPStatus, message: str, *, param: str | None = None
        ) -> None:
            error: dict[str, Any] = {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
            self._reply(status, {"error": error})

        def _begin_sse(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        def _write_sse(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + encoded + b"\n\n")
            self.wfile.flush()

        def _end_sse(self) -> None:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            # The stream is finite. Closing after [DONE] avoids clients waiting
            # indefinitely for a keep-alive SSE connection.
            self.close_connection = True

        def _stream_chat_completion(self, request: dict[str, Any]) -> None:
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            self._begin_sse()
            try:
                self._write_sse(
                    openai_completion_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=request["model"],
                        delta={"role": "assistant", "content": ""},
                        finish_reason=None,
                    )
                )

                def on_decode_token(delta: str) -> None:
                    self._write_sse(
                        openai_completion_chunk(
                            completion_id=completion_id,
                            created=created,
                            model=request["model"],
                            delta={"content": delta},
                            finish_reason=None,
                        )
                    )

                response = service.chat_completion_from_request(
                    request,
                    on_decode_token=on_decode_token,
                    completion_id=completion_id,
                    created=created,
                )
                self._write_sse(
                    openai_completion_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=request["model"],
                        delta={},
                        finish_reason=response["choices"][0]["finish_reason"],
                    )
                )
            except (BrokenPipeError, ConnectionResetError):
                # The client disconnected; the decode callback must not turn
                # that into an unrelated server-side traceback.
                return
            except Exception as error:  # Headers are already committed.
                try:
                    self._write_sse(
                        {
                            "error": {
                                "message": str(error),
                                "type": "server_error",
                                "param": None,
                                "code": None,
                            }
                        }
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                try:
                    self._end_sse()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > service.max_request_bytes:
                raise ValueError(
                    f"request body must be JSON below {service.max_request_bytes} bytes"
                )
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON request must be an object")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/health":
                self._reply(HTTPStatus.OK, service.health())
                return
            if path == "/v1/models":
                self._reply(HTTPStatus.OK, service.openai_models())
                return
            if path == f"/v1/models/{service.openai_model_id}":
                self._reply(HTTPStatus.OK, service.openai_models()["data"][0])
                return
            self._reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path not in ("/v1/generate", "/v1/chat/completions"):
                self._reply(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                payload = self._read_json_body()
                if path == "/v1/generate":
                    self._reply(HTTPStatus.OK, service.generate(payload))
                    return
                request = parse_openai_chat_request(
                    payload, service.openai_model_id, service.max_new_tokens
                )
                if request["stream"]:
                    self._stream_chat_completion(request)
                    return
                response = service.chat_completion_from_request(request)
                self._reply(HTTPStatus.OK, response)
            except (ValueError, FileNotFoundError) as error:
                if path == "/v1/chat/completions":
                    self._reply_openai_error(HTTPStatus.BAD_REQUEST, str(error))
                else:
                    self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except subprocess.TimeoutExpired:
                if path == "/v1/chat/completions":
                    self._reply_openai_error(HTTPStatus.GATEWAY_TIMEOUT, "pipeline timed out")
                else:
                    self._reply(HTTPStatus.GATEWAY_TIMEOUT, {"error": "pipeline timed out"})
            except Exception as error:  # pragma: no cover - preserves diagnostic log
                if path == "/v1/chat/completions":
                    self._reply(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "error": {
                                "message": str(error),
                                "type": "server_error",
                                "param": None,
                                "code": None,
                            }
                        },
                    )
                else:
                    self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            print("[http] " + format % args, flush=True)

    return Handler


def main() -> None:
    args = parse_args()
    service = GelabService(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(service))
    print(
        json.dumps(
            {
                "listen": f"http://{args.host}:{args.port}",
                "health": f"http://{args.host}:{args.port}/health",
                "generate": f"http://{args.host}:{args.port}/v1/generate",
                "openai_models": f"http://{args.host}:{args.port}/v1/models",
                "openai_chat_completions": f"http://{args.host}:{args.port}/v1/chat/completions",
                "openai_model_id": service.openai_model_id,
                "vision_executor": service.vision_executor,
                "prefill_executor": service.prefill_executor,
                "tfdl_runtime": service.runtime_startup,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
