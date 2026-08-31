"""Python API for the persistent ``llmdecode --serve`` native worker.

The worker owns a mmap'ed GGUF model for its complete lifetime.  Individual
requests provide only a temporary descriptor plus FP16 KV/logit NPY files.
This makes it possible to consume TFDL prefill cache directly, without ONNX
decoder sessions or a checkpoint, while retaining token-level streaming.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Callable, Sequence, TextIO

import numpy as np


class ExternalKvWorkerError(RuntimeError):
    """The native llama.cpp worker rejected a request or stopped."""


def packaged_binary() -> Path:
    """Return the architecture-specific binary installed in this wheel."""
    binary = Path(__file__).resolve().parent / "bin" / "llmdecode"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            "the tfdl-llmdecode wheel has no executable llmdecode binary; "
            "build the target-architecture wheel with scripts/build_wheel_arm.sh"
        )
    return binary


class ExternalKvDecodeWorker:
    """One serial, model-resident llama.cpp external-KV decode worker.

    ``prompt_tokens`` is deliberately an argument to :meth:`generate`, not a
    construction option.  It is the live request length.  The prefill bucket
    may be 256, but only the valid prefix is permitted in each KV tensor.
    """

    def __init__(
        self,
        *,
        model: str | Path,
        threads: int = 0,
        kv_cache_type: str = "fp16",
        binary: str | Path | None = None,
        startup_timeout_seconds: float = 180.0,
    ) -> None:
        self.model = Path(model).expanduser().resolve()
        if not self.model.is_file():
            raise FileNotFoundError(f"GGUF decoder model is unavailable: {self.model}")
        if kv_cache_type not in {"fp16", "q8_0"}:
            raise ValueError("kv_cache_type must be fp16 or q8_0")
        if threads < 0:
            raise ValueError("threads must be non-negative")
        self.binary = Path(binary).expanduser().resolve() if binary else packaged_binary()
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"llmdecode binary is unavailable: {self.binary}")
        self.threads = int(threads)
        self.kv_cache_type = kv_cache_type
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._startup_stderr: TextIO | None = None
        self._startup_seconds = 0.0
        self._start(startup_timeout_seconds)

    @property
    def startup_seconds(self) -> float:
        return self._startup_seconds

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _start(self, timeout_seconds: float) -> None:
        started = time.perf_counter()
        command = [
            str(self.binary),
            "--serve",
            "--model",
            str(self.model),
            "--kv-cache-type",
            self.kv_cache_type,
        ]
        if self.threads:
            command.extend(["--threads", str(self.threads)])
        # llama.cpp emits model-load diagnostics to stderr.  A PIPE would
        # deadlock the worker before it prints its JSON ready event once that
        # pipe fills.  A temporary file retains the failure diagnostics while
        # allowing unbounded native startup logs.
        self._startup_stderr = tempfile.TemporaryFile(
            mode="w+t", encoding="utf-8", errors="replace"
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._startup_stderr,
            text=True,
            bufsize=1,
        )
        if process.stdout is None or process.stdin is None:
            process.kill()
            self._close_startup_stderr()
            raise ExternalKvWorkerError("could not create llmdecode worker pipes")
        # The native worker emits ready only after GGUF mapping/model loading.
        # ``readline`` is intentionally blocking here: API startup must make
        # decoder readiness visible before accepting the first request.
        ready = process.stdout.readline()
        if not ready:
            process.wait(timeout=min(timeout_seconds, 5.0))
            diagnostic = self._startup_stderr_tail()
            self._close_startup_stderr()
            suffix = f": {diagnostic[-4000:]}" if diagnostic else ""
            raise ExternalKvWorkerError(
                f"llmdecode worker stopped during startup (exit {process.returncode}){suffix}"
            )
        try:
            payload = json.loads(ready)
        except json.JSONDecodeError as error:
            process.kill()
            self._close_startup_stderr()
            raise ExternalKvWorkerError(f"invalid llmdecode startup response: {ready!r}") from error
        if payload.get("event") != "ready":
            process.kill()
            self._close_startup_stderr()
            raise ExternalKvWorkerError(f"llmdecode failed startup: {payload}")
        self._close_startup_stderr()
        self._process = process
        self._startup_seconds = time.perf_counter() - started

    def _startup_stderr_tail(self) -> str:
        stream = self._startup_stderr
        if stream is None:
            return ""
        stream.flush()
        stream.seek(0)
        return stream.read()

    def _close_startup_stderr(self) -> None:
        stream, self._startup_stderr = self._startup_stderr, None
        if stream is not None:
            stream.close()

    @staticmethod
    def _ensure_path(value: str | Path) -> str:
        path = str(Path(value).resolve())
        if "\n" in path or "\r" in path:
            raise ValueError("worker descriptor paths must not contain newlines")
        return path

    def generate(
        self,
        *,
        logits: str | Path,
        positions: str | Path,
        keys: Sequence[str | Path],
        values: Sequence[str | Path],
        kv_heads: int,
        head_dim: int,
        prompt_tokens: int,
        first_decode_position: int,
        max_new_tokens: int,
        descriptor: str | Path,
        on_token_id: Callable[[int], None] | None = None,
    ) -> dict[str, object]:
        """Run one request and return the native report.

        Caller-owned KV files must already be C-contiguous FP16
        ``[1, Hkv, prompt_tokens, D]``.  This exact-shape rule makes padded
        cache leakage fail fast rather than silently changing an answer.
        """
        if prompt_tokens <= 0 or max_new_tokens <= 0:
            raise ValueError("prompt_tokens and max_new_tokens must be positive")
        if first_decode_position < 0:
            raise ValueError("first_decode_position must be non-negative")
        if len(keys) != len(values) or not keys:
            raise ValueError("keys and values must have the same non-zero layer count")
        lines = [
            f"logits={self._ensure_path(logits)}",
            f"positions={self._ensure_path(positions)}",
            f"layers={len(keys)}",
            f"kv_heads={int(kv_heads)}",
            f"head_dim={int(head_dim)}",
            f"prompt_tokens={int(prompt_tokens)}",
            f"first_decode_position={int(first_decode_position)}",
            f"max_new_tokens={int(max_new_tokens)}",
        ]
        for key, value in zip(keys, values):
            lines.append(f"key={self._ensure_path(key)}")
            lines.append(f"value={self._ensure_path(value)}")
        descriptor_path = Path(descriptor).resolve()
        descriptor_path.write_text("\n".join(lines) + "\n")

        with self._lock:
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise ExternalKvWorkerError("llmdecode worker is closed")
            if process.poll() is not None:
                raise ExternalKvWorkerError(f"llmdecode worker exited with {process.returncode}")
            process.stdin.write(str(descriptor_path) + "\n")
            process.stdin.flush()
            while True:
                line = process.stdout.readline()
                if not line:
                    raise ExternalKvWorkerError(
                        f"llmdecode worker exited with {process.poll()} during decode"
                    )
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ExternalKvWorkerError(f"invalid llmdecode worker response: {line!r}") from error
                kind = event.get("event")
                if kind == "token":
                    token = event.get("token_id")
                    if not isinstance(token, int):
                        raise ExternalKvWorkerError("worker emitted token event without an integer ID")
                    if on_token_id is not None:
                        on_token_id(token)
                elif kind == "result":
                    result = event.get("result")
                    if not isinstance(result, dict):
                        raise ExternalKvWorkerError("worker result event has no result object")
                    return result
                elif kind == "error":
                    raise ExternalKvWorkerError(str(event.get("message", "unknown native error")))
                else:
                    raise ExternalKvWorkerError(f"unknown worker event: {event}")

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                return
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                    process.wait(timeout=10)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.terminate()
            if process.poll() is None:
                process.kill()

    def __enter__(self) -> "ExternalKvDecodeWorker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
