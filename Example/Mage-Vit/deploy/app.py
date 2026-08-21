#!/usr/bin/env python3
"""Flask API for video -> NPU vision/prefill -> ORT W8A8 decode."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
SDK_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CONFIG = DEPLOY_DIR / "deployment.json"
ALLOWED_VIDEO_SUFFIXES = {
    ".avi",
    ".h264",
    ".h265",
    ".hevc",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}
STREAM_PREFIX = "MEGAVIT_EVENT "


class DeploymentError(RuntimeError):
    """An expected deployment or request failure."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class DeploymentConfig:
    config_path: Path
    profile: str
    model_path: Path
    frontend_bin: Path
    vision_fb: Path
    addon: Path
    executor_config: Path
    qwen_fb_dir: Path
    decoder_dir: Path
    work_dir: Path
    seq_len: int
    target_canvases: int
    vision_workers: int
    outlier_top_k: int
    question: str
    system: str
    max_question_tokens: int
    max_new_tokens: int
    max_allowed_new_tokens: int
    threads: int
    request_timeout_seconds: int
    max_upload_bytes: int
    require_npu: bool
    keep_jobs: bool
    persistent_runtime: bool

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DeploymentConfig":
        config_path = Path(
            path or os.environ.get("MEGAVIT_DEPLOY_CONFIG", DEFAULT_CONFIG)
        ).resolve()
        raw = json.loads(config_path.read_text())
        if raw.get("format") != "mage-vl-deployment-v1":
            raise ValueError(f"unsupported deployment config: {config_path}")
        root = config_path.parent.parent
        bucket = raw["prefill_bucket"]
        model_value = os.environ.get(
            "MAGE_VL_MODEL_PATH", raw.get("model_path", "")
        )
        frontend_value = os.environ.get(
            "MEGAVIT_FRONTEND_BIN", raw["frontend_bin"]
        )
        work_value = os.environ.get("MEGAVIT_WORK_DIR", raw["work_dir"])
        model_path = (
            _resolve(root, model_value)
            if model_value
            else Path("/__MAGE_VL_MODEL_PATH_NOT_SET__")
        )
        return cls(
            config_path=config_path,
            profile=str(raw["profile"]),
            model_path=model_path,
            frontend_bin=_resolve(root, frontend_value),
            vision_fb=_resolve(root, raw["vision_fb"]),
            addon=_resolve(root, raw["addon"]),
            executor_config=_resolve(root, raw["executor_config"]),
            qwen_fb_dir=_resolve(root, bucket["fb_dir"]),
            decoder_dir=_resolve(root, raw["decoder_dir"]),
            work_dir=_resolve(root, work_value),
            seq_len=int(bucket["seq_len"]),
            target_canvases=int(bucket["target_canvases"]),
            vision_workers=int(
                os.environ.get(
                    "MEGAVIT_VISION_WORKERS", raw.get("vision_workers", 1)
                )
            ),
            outlier_top_k=int(bucket.get("outlier_top_k", 4)),
            question=str(bucket["question"]),
            system=str(bucket["system"]),
            max_question_tokens=int(bucket["max_question_tokens"]),
            max_new_tokens=int(raw["max_new_tokens"]),
            max_allowed_new_tokens=int(raw["max_allowed_new_tokens"]),
            threads=int(raw["threads"]),
            request_timeout_seconds=int(raw["request_timeout_seconds"]),
            max_upload_bytes=int(raw["max_upload_bytes"]),
            require_npu=bool(raw.get("require_npu", True)),
            keep_jobs=bool(raw.get("keep_jobs", False)),
            persistent_runtime=bool(raw.get("persistent_runtime", False)),
        )


class MegaVitService:
    """Serializes access to the NPU and owns deployment validation."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self._inference_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._tokenizer: Any | None = None
        self._last_validation: dict[str, Any] | None = None
        self._runtime: Any | None = None
        self._runtime_error: str | None = None

    def initialize(self) -> None:
        """Construct all inference objects once for this Flask worker."""
        if not self.config.persistent_runtime or self._runtime is not None:
            return
        try:
            try:
                from .persistent_runtime import PersistentMageRuntime
            except ImportError:
                from persistent_runtime import PersistentMageRuntime

            self._runtime = PersistentMageRuntime(self.config)
            # Prompt validation and inference now share one tokenizer object.
            self._tokenizer = self._runtime.tokenizer
            self._tokenizer_lock = self._runtime.tokenizer_lock
            self._runtime_error = None
        except Exception as error:
            self._runtime_error = f"{type(error).__name__}: {error}"
            raise

    def _frontend_capabilities(self) -> tuple[dict[str, bool], str | None]:
        if not self.config.frontend_bin.is_file():
            return {}, "frontend binary is missing"
        if not os.access(self.config.frontend_bin, os.X_OK):
            return {}, "frontend binary is not executable"
        try:
            completed = subprocess.run(
                [str(self.config.frontend_bin), "--capabilities"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if completed.returncode:
                return {}, completed.stderr.strip() or "capability probe failed"
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                return {}, "capability probe produced no output"
            for line in reversed(lines):
                if not line.startswith("{"):
                    continue
                value = json.loads(line)
                if "ffmpeg" in value and "tfdl" in value:
                    return value, None
            return {}, "capability probe did not emit its JSON record"
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            return {}, str(error)

    def _python_tfdl_runtime(self) -> tuple[dict[str, Any], str | None]:
        """Probe the exact Python/addon pair used by the prefill subprocess."""
        if not self.config.addon.is_file():
            return {}, "addon is missing"
        probe = (
            "import json, sys; "
            "from TFDL2.utils import LoadCustomOp; "
            "import TFDL2, TFDL2.TFDL2 as native; "
            "result=LoadCustomOp(sys.argv[1]); "
            "print(json.dumps({'package': TFDL2.__file__, "
            "'native': native.__file__, 'addon_result': result}))"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe, str(self.config.addon)],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
                env=self._runtime_environment(),
            )
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip()
                return {}, detail or "TFDL2 Python/addon probe failed"
            for line in reversed(completed.stdout.splitlines()):
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict) and "native" in value:
                    return value, None
            return {}, "TFDL2 Python/addon probe emitted no JSON record"
        except (OSError, subprocess.TimeoutExpired) as error:
            return {}, str(error)

    def validate(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._last_validation is not None and not refresh:
            return self._last_validation
        cfg = self.config
        issues: list[str] = []
        warnings: list[str] = []
        if cfg.max_question_tokens <= 0:
            issues.append("max_question_tokens must be positive")
        if cfg.vision_workers <= 0:
            issues.append("vision_workers must be positive")
        required_files = {
            "model config": cfg.model_path / "config.json",
            "model weight index": cfg.model_path / "model.safetensors.index.json",
            "vision FB": cfg.vision_fb,
            "addon": cfg.addon,
            "executor config": cfg.executor_config,
            "prefill manifest": cfg.qwen_fb_dir / "manifest.json",
            "decoder manifest": cfg.decoder_dir / "manifest.json",
            "decoder model": cfg.decoder_dir / "decoder.w8a8.onnx",
            "decoder final head": cfg.decoder_dir / "final_head.w8a8.onnx",
        }
        for label, path in required_files.items():
            if not path.is_file():
                issues.append(f"missing {label}: {path}")
        index_path = cfg.model_path / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text())
                shards = sorted(set(index["weight_map"].values()))
                missing_shards = [
                    name for name in shards if not (cfg.model_path / name).is_file()
                ]
                if missing_shards:
                    issues.append(f"missing checkpoint shards: {missing_shards}")
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"invalid model weight index: {error}")
        prefill_manifest_path = cfg.qwen_fb_dir / "manifest.json"
        if prefill_manifest_path.is_file():
            try:
                manifest = json.loads(prefill_manifest_path.read_text())
                expected = {
                    "seq_len": cfg.seq_len,
                    "attention_mode": "arm-causal-hxs",
                    "outlier_top_k": cfg.outlier_top_k,
                    "token_hybrid_qkv_start_layer": 12,
                }
                for name, value in expected.items():
                    if manifest.get(name) != value:
                        issues.append(
                            f"prefill manifest {name}={manifest.get(name)!r}, "
                            f"expected {value!r}"
                        )
                if manifest.get("layers") != list(range(36)):
                    issues.append("prefill manifest does not contain ordered layers 0..35")
                reports = manifest.get("reports", [])
                tensor_audits = [
                    report.get("tensor_audit", {})
                    for report in reports
                    if isinstance(report, dict)
                ]
                if (
                    len(tensor_audits) != 36
                    or any(
                        audit.get("ok") is not True
                        or audit.get("invalid_int8_qinfo_count") != 0
                        for audit in tensor_audits
                    )
                ):
                    issues.append(
                        "prefill manifest lacks clean all-tensor INT8 qinfo audits"
                    )
                if manifest.get("token_group_boundaries") != [21]:
                    issues.append("prefill token-group boundaries mismatch")
                calibration = manifest.get("calibration", {})
                valid_lengths = calibration.get("valid_seq_lens", [])
                if (
                    not calibration.get("padding_ranges_ignored", False)
                    or not calibration.get("causal_qk_cells_only", False)
                    or cfg.seq_len not in valid_lengths
                ):
                    issues.append(
                        "prefill manifest lacks complete flexible-bucket "
                        "calibration provenance"
                    )
            except (TypeError, ValueError) as error:
                issues.append(f"invalid prefill manifest: {error}")
        decoder_manifest_path = cfg.decoder_dir / "manifest.json"
        if decoder_manifest_path.is_file():
            try:
                manifest = json.loads(decoder_manifest_path.read_text())
                if manifest.get("format") != "mage-qwen3-ort-decoder-v1":
                    issues.append("decoder manifest format mismatch")
                if manifest.get("model_file") != "decoder.w8a8.onnx":
                    issues.append("decoder manifest model_file mismatch")
                if manifest.get("final_head_model") != "final_head.w8a8.onnx":
                    issues.append("decoder manifest final_head_model mismatch")
                if int(manifest["config"]["num_hidden_layers"]) != 36:
                    issues.append("decoder manifest does not contain 36 layers")
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"invalid decoder manifest: {error}")
        missing_layers = [
            layer
            for layer in range(36)
            if not (
                cfg.qwen_fb_dir / f"layer_{layer:02d}_seq_{cfg.seq_len}.fb"
            ).is_file()
        ]
        if missing_layers:
            issues.append(f"missing prefill FB layers: {missing_layers}")
        decoder_names = ["final_head"] + [
            f"layer_{layer:02d}" for layer in range(36)
        ]
        missing_decoder_data = [
            name
            for name in decoder_names
            if not (
                cfg.decoder_dir / "layers" / f"{name}.w8a8.onnx.data"
            ).is_file()
        ]
        if missing_decoder_data:
            issues.append(
                "missing ORT external data: " + ", ".join(missing_decoder_data)
            )
        capabilities, capability_error = self._frontend_capabilities()
        if capability_error:
            issues.append(f"frontend capability check: {capability_error}")
        else:
            if not capabilities.get("ffmpeg", False):
                issues.append("frontend was built without FFmpeg support")
            if not capabilities.get("tfdl", False):
                issues.append("frontend was built without TFDL support")
            if capabilities.get("ffmpeg", False) and not capabilities.get(
                "patched_bitcost", False
            ):
                warnings.append(
                    "patched FFmpeg bitcost shim is absent; using the "
                    "motion-vector/pixel-residual fallback"
                )
        python_tfdl, python_tfdl_error = self._python_tfdl_runtime()
        if python_tfdl_error:
            issues.append(
                "Python TFDL2/addon ABI check failed: " + python_tfdl_error
            )
        npu_devices = sorted(glob.glob("/dev/thinkforce*"))
        if cfg.require_npu and not npu_devices:
            issues.append("no /dev/thinkforce* NPU device is visible")
        result = {
            "ready": not issues,
            "profile": cfg.profile,
            "seq_len": cfg.seq_len,
            "target_canvases": cfg.target_canvases,
            "vision_workers": cfg.vision_workers,
            "busy": self._inference_lock.locked(),
            "npu_devices": npu_devices,
            "frontend_capabilities": capabilities,
            "python_tfdl": python_tfdl,
            "issues": issues,
            "warnings": warnings,
            "persistent_runtime": {
                "enabled": cfg.persistent_runtime,
                "initialized": self._runtime is not None,
                "error": self._runtime_error,
                "startup": (
                    self._runtime.startup
                    if self._runtime is not None else None
                ),
            },
        }
        self._last_validation = result
        return result

    def question_token_count(self, question: str) -> int:
        """Cheap request preflight before codec and NPU work starts."""
        value = question.strip()
        if not value:
            raise DeploymentError("question must not be empty", 422)
        with self._tokenizer_lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path, trust_remote_code=True
                )
            tokens = self._tokenizer(
                value, add_special_tokens=False
            ).input_ids
        count = len(tokens)
        if count > self.config.max_question_tokens:
            raise DeploymentError(
                f"question contains {count} tokens; this S={self.config.seq_len} "
                "bucket allows at most "
                f"{self.config.max_question_tokens} question tokens",
                422,
            )
        return count

    def build_command(
        self,
        video: Path,
        output: Path,
        *,
        question: str,
        max_new_tokens: int,
        stream_jsonl: bool = False,
    ) -> list[str]:
        cfg = self.config
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_ort_pipeline.py"),
            "--model-path",
            str(cfg.model_path),
            "--video",
            str(video),
            "--frontend-bin",
            str(cfg.frontend_bin),
            "--vision-fb",
            str(cfg.vision_fb),
            "--target-canvases",
            str(cfg.target_canvases),
            "--expected-seq-len",
            str(cfg.seq_len),
            "--qwen-fb-dir",
            str(cfg.qwen_fb_dir),
            "--decoder-dir",
            str(cfg.decoder_dir),
            "--output-dir",
            str(output),
            "--question",
            question,
            "--system",
            cfg.system,
            "--max-new-tokens",
            str(max_new_tokens),
            "--threads",
            str(cfg.threads),
            "--hardware",
            "--addon",
            str(cfg.addon),
            "--executor-config",
            str(cfg.executor_config),
        ]
        if stream_jsonl:
            command.append("--stream-jsonl")
        return command

    def _runtime_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        library_paths = [
            str(SDK_ROOT / "lib"),
            str(self.config.addon.parent),
        ]
        existing = environment.get("LD_LIBRARY_PATH")
        if existing:
            library_paths.append(existing)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
        return environment

    def _validate_generate_request(
        self, question: str, max_new_tokens: int
    ) -> tuple[str, int]:
        # Asset/ABI validation is stable for the lifetime of a resident worker.
        # `/health` remains the explicit refresh path; inference requests use
        # the cached result and avoid spawning validation probes every time.
        health = self.validate(refresh=False)
        if not health["ready"]:
            raise DeploymentError("deployment is not ready", 503)
        normalized = question.strip()
        question_tokens = self.question_token_count(normalized)
        if not 1 <= max_new_tokens <= self.config.max_allowed_new_tokens:
            raise DeploymentError(
                "max_new_tokens must be inside [1, "
                f"{self.config.max_allowed_new_tokens}]",
                422,
            )
        return normalized, question_tokens

    @staticmethod
    def _video_suffix(uploaded: Any) -> str:
        filename = secure_filename(uploaded.filename or "video.mp4")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise DeploymentError(
                f"unsupported video extension {suffix or '<none>'}", 415
            )
        return suffix

    @staticmethod
    def _log_tail(path: Path, limit: int = 12000) -> str:
        if not path.exists():
            return ""
        data = path.read_bytes()
        return data[-limit:].decode("utf-8", errors="replace")

    @staticmethod
    def _append_job_log(path: Path, value: str) -> None:
        with path.open("a", encoding="utf-8") as log:
            log.write(value)
            if value and not value.endswith("\n"):
                log.write("\n")

    def _persistent_events(
        self,
        uploaded: Any,
        *,
        question: str,
        max_new_tokens: int,
    ) -> Iterator[dict[str, Any]]:
        """Prepare a request synchronously, then stream in-process events."""
        if self._runtime is None:
            raise DeploymentError("persistent runtime is not initialized", 503)
        question, question_tokens = self._validate_generate_request(
            question, max_new_tokens
        )
        suffix = self._video_suffix(uploaded)
        if not self._inference_lock.acquire(blocking=False):
            raise DeploymentError("NPU is busy with another request", 429)
        job_id = uuid.uuid4().hex
        job_root = self.config.work_dir / job_id
        output = job_root / "output"
        log_path = job_root / "pipeline.log"
        try:
            job_root.mkdir(parents=True, exist_ok=False)
            video = job_root / ("input" + suffix)
            uploaded.save(video)
            if video.stat().st_size == 0:
                raise DeploymentError("uploaded video is empty", 400)
        except Exception:
            self._inference_lock.release()
            shutil.rmtree(job_root, ignore_errors=True)
            raise

        def event_stream() -> Iterator[dict[str, Any]]:
            inference_started = time.perf_counter()
            first_token_at: float | None = None
            try:
                yield {
                    "type": "accepted",
                    "job_id": job_id,
                    "profile": self.config.profile,
                    "question": question,
                    "question_tokens": question_tokens,
                    "persistent_runtime": True,
                }
                runtime_stream = self._runtime.stream(
                    video,
                    output,
                    question=question,
                    system=self.config.system,
                    max_new_tokens=max_new_tokens,
                    timeout_seconds=self.config.request_timeout_seconds,
                    log_path=log_path,
                )
                for event in runtime_stream:
                    self._append_job_log(
                        log_path,
                        json.dumps(event, ensure_ascii=False),
                    )
                    now = time.perf_counter()
                    if (
                        event.get("type") == "token"
                        and event.get("text")
                        and first_token_at is None
                    ):
                        first_token_at = now
                        event["time_to_first_token_seconds"] = (
                            now - inference_started
                        )
                    if event.get("type") != "pipeline_done":
                        yield event
                        continue
                    decode = event["decode_report"]
                    total_seconds = now - inference_started
                    yield {
                        "type": "done",
                        "job_id": job_id,
                        "text": decode["text"],
                        "tokens": decode["tokens"],
                        "prompt_tokens": decode["prompt_tokens"],
                        "generated_tokens": decode["generated_tokens"],
                        "time_to_first_token_seconds": (
                            first_token_at - inference_started
                            if first_token_at is not None else None
                        ),
                        "decode_seconds": (
                            now - first_token_at
                            if first_token_at is not None else None
                        ),
                        "total_seconds": total_seconds,
                        "ort_tokens_per_second": decode[
                            "ort_tokens_per_second"
                        ],
                        "ort_session_load_seconds": decode[
                            "session_load_seconds"
                        ],
                        "ort_persistent_session_load_seconds": decode[
                            "persistent_session_load_seconds"
                        ],
                        "ort_decode_loop_seconds": decode[
                            "decode_loop_seconds"
                        ],
                        "stages": event["stages"],
                        "persistent_runtime": True,
                    }
            except GeneratorExit:
                raise
            except Exception as error:
                self._append_job_log(log_path, traceback.format_exc())
                message = str(error)
                status = 504 if isinstance(
                    error, (TimeoutError, subprocess.TimeoutExpired)
                ) else (422 if "bucket supports at most" in message else 500)
                yield {
                    "type": "error",
                    "status": status,
                    "job_id": job_id,
                    "message": message,
                    "log_tail": self._log_tail(log_path),
                }
            finally:
                self._inference_lock.release()
                if not self.config.keep_jobs and job_root.exists():
                    shutil.rmtree(job_root, ignore_errors=True)

        return event_stream()

    def generate(
        self,
        uploaded: Any,
        *,
        question: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if self._runtime is not None:
            final: dict[str, Any] | None = None
            events = self._persistent_events(
                uploaded,
                question=question,
                max_new_tokens=max_new_tokens,
            )
            try:
                for event in events:
                    if event["type"] == "error":
                        raise DeploymentError(
                            str(event["message"]),
                            int(event.get("status", 500)),
                        )
                    if event["type"] == "done":
                        final = event
            finally:
                close = getattr(events, "close", None)
                if close is not None:
                    close()
            if final is None:
                raise DeploymentError("persistent runtime returned no result")
            return {
                "job_id": final["job_id"],
                "profile": self.config.profile,
                "question": question.strip(),
                "question_tokens": self.question_token_count(question),
                "text": final["text"],
                "tokens": final["tokens"],
                "prompt_tokens": final["prompt_tokens"],
                "generated_tokens": final["generated_tokens"],
                "ort_tokens_per_second": final["ort_tokens_per_second"],
                "wall_seconds": final["total_seconds"],
                "stages": final["stages"],
                "persistent_runtime": True,
            }
        question, question_tokens = self._validate_generate_request(
            question, max_new_tokens
        )
        suffix = self._video_suffix(uploaded)
        if not self._inference_lock.acquire(blocking=False):
            raise DeploymentError("NPU is busy with another request", 429)
        job_id = uuid.uuid4().hex
        job_root = self.config.work_dir / job_id
        output = job_root / "output"
        log_path = job_root / "pipeline.log"
        try:
            job_root.mkdir(parents=True, exist_ok=False)
            video = job_root / ("input" + suffix)
            uploaded.save(video)
            if video.stat().st_size == 0:
                raise DeploymentError("uploaded video is empty", 400)
            command = self.build_command(
                video,
                output,
                question=question,
                max_new_tokens=max_new_tokens,
            )
            started = time.perf_counter()
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._runtime_environment(),
                    timeout=self.config.request_timeout_seconds,
                )
            wall_seconds = time.perf_counter() - started
            if completed.returncode:
                tail = self._log_tail(log_path)
                if "bucket supports at most" in tail:
                    raise DeploymentError(
                        "the complete video/question prompt exceeds the "
                        f"S={self.config.seq_len} bucket",
                        422,
                    )
                raise DeploymentError(
                    "pipeline failed with return code "
                    f"{completed.returncode}: {tail}"
                )
            decode = json.loads((output / "decode.report.json").read_text())
            pipeline = json.loads((output / "pipeline.report.json").read_text())
            return {
                "job_id": job_id,
                "profile": self.config.profile,
                "question": question,
                "question_tokens": question_tokens,
                "text": decode["text"],
                "tokens": decode["tokens"],
                "prompt_tokens": decode["prompt_tokens"],
                "generated_tokens": decode["generated_tokens"],
                "ort_tokens_per_second": decode["ort_tokens_per_second"],
                "wall_seconds": wall_seconds,
                "stages": pipeline["stages"],
            }
        except subprocess.TimeoutExpired as error:
            raise DeploymentError(
                f"pipeline exceeded {self.config.request_timeout_seconds}s timeout"
            ) from error
        finally:
            self._inference_lock.release()
            if not self.config.keep_jobs and job_root.exists():
                shutil.rmtree(job_root, ignore_errors=True)

    def stream_generate(
        self,
        uploaded: Any,
        *,
        question: str,
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Run one job and forward token/stage events as NDJSON."""
        if self._runtime is not None:
            events = self._persistent_events(
                uploaded,
                question=question,
                max_new_tokens=max_new_tokens,
            )

            def persistent_json_stream() -> Iterator[str]:
                try:
                    for event in events:
                        yield json.dumps(event, ensure_ascii=False) + "\n"
                finally:
                    close = getattr(events, "close", None)
                    if close is not None:
                        close()

            return persistent_json_stream()
        question, question_tokens = self._validate_generate_request(
            question, max_new_tokens
        )
        suffix = self._video_suffix(uploaded)
        if not self._inference_lock.acquire(blocking=False):
            raise DeploymentError("NPU is busy with another request", 429)
        job_id = uuid.uuid4().hex
        job_root = self.config.work_dir / job_id
        output = job_root / "output"
        log_path = job_root / "pipeline.log"
        try:
            job_root.mkdir(parents=True, exist_ok=False)
            video = job_root / ("input" + suffix)
            uploaded.save(video)
            if video.stat().st_size == 0:
                raise DeploymentError("uploaded video is empty", 400)
        except Exception:
            self._inference_lock.release()
            shutil.rmtree(job_root, ignore_errors=True)
            raise

        command = self.build_command(
            video,
            output,
            question=question,
            max_new_tokens=max_new_tokens,
            stream_jsonl=True,
        )

        def event_stream() -> Iterator[str]:
            process: subprocess.Popen[str] | None = None
            timeout_timer: threading.Timer | None = None
            kill_timer: threading.Timer | None = None
            timed_out = threading.Event()
            inference_started = time.perf_counter()
            first_token_at: float | None = None
            try:
                with log_path.open("w", encoding="utf-8", buffering=1) as log:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        env=self._runtime_environment(),
                    )

                    def stop_timed_out_process() -> None:
                        nonlocal kill_timer
                        if process is None or process.poll() is not None:
                            return
                        timed_out.set()
                        try:
                            process.terminate()
                        except OSError:
                            return

                        def force_kill() -> None:
                            if process is not None and process.poll() is None:
                                try:
                                    process.kill()
                                except OSError:
                                    pass

                        kill_timer = threading.Timer(5.0, force_kill)
                        kill_timer.daemon = True
                        kill_timer.start()

                    timeout_timer = threading.Timer(
                        self.config.request_timeout_seconds,
                        stop_timed_out_process,
                    )
                    timeout_timer.daemon = True
                    timeout_timer.start()
                    yield json.dumps(
                        {
                            "type": "accepted",
                            "job_id": job_id,
                            "profile": self.config.profile,
                            "question": question,
                            "question_tokens": question_tokens,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                    assert process.stdout is not None
                    for line in process.stdout:
                        log.write(line)
                        marker = line.find(STREAM_PREFIX)
                        if marker < 0:
                            continue
                        try:
                            event = json.loads(
                                line[marker + len(STREAM_PREFIX) :]
                            )
                        except ValueError:
                            continue
                        now = time.perf_counter()
                        if (
                            event.get("type") == "token"
                            and event.get("text")
                            and first_token_at is None
                        ):
                            first_token_at = now
                            event["time_to_first_token_seconds"] = (
                                now - inference_started
                            )
                        yield json.dumps(event, ensure_ascii=False) + "\n"
                    returncode = process.wait()
                inference_finished = time.perf_counter()
                total_seconds = inference_finished - inference_started
                if timed_out.is_set():
                    yield json.dumps(
                        {
                            "type": "error",
                            "status": 504,
                            "job_id": job_id,
                            "message": "pipeline exceeded "
                            f"{self.config.request_timeout_seconds}s timeout",
                            "log_tail": self._log_tail(log_path),
                        }
                    ) + "\n"
                    return
                if returncode:
                    tail = self._log_tail(log_path)
                    status = 422 if "bucket supports at most" in tail else 500
                    yield json.dumps(
                        {
                            "type": "error",
                            "status": status,
                            "job_id": job_id,
                            "message": (
                                "the complete video/question prompt exceeds "
                                f"the S={self.config.seq_len} bucket"
                                if status == 422
                                else f"pipeline failed with return code {returncode}"
                            ),
                            "log_tail": tail,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                    return
                decode = json.loads((output / "decode.report.json").read_text())
                pipeline = json.loads(
                    (output / "pipeline.report.json").read_text()
                )
                yield json.dumps(
                    {
                        "type": "done",
                        "job_id": job_id,
                        "text": decode["text"],
                        "tokens": decode["tokens"],
                        "prompt_tokens": decode["prompt_tokens"],
                        "generated_tokens": decode["generated_tokens"],
                        "time_to_first_token_seconds": (
                            first_token_at - inference_started
                            if first_token_at is not None
                            else None
                        ),
                        "decode_seconds": (
                            inference_finished - first_token_at
                            if first_token_at is not None
                            else None
                        ),
                        "total_seconds": total_seconds,
                        "ort_tokens_per_second": decode[
                            "ort_tokens_per_second"
                        ],
                        "ort_session_load_seconds": decode[
                            "session_load_seconds"
                        ],
                        "ort_decode_loop_seconds": decode[
                            "decode_loop_seconds"
                        ],
                        "stages": pipeline["stages"],
                    },
                    ensure_ascii=False,
                ) + "\n"
            except GeneratorExit:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            except Exception as error:
                yield json.dumps(
                    {
                        "type": "error",
                        "status": 500,
                        "job_id": job_id,
                        "message": str(error),
                        "log_tail": self._log_tail(log_path),
                    },
                    ensure_ascii=False,
                ) + "\n"
            finally:
                if timeout_timer is not None:
                    timeout_timer.cancel()
                if kill_timer is not None:
                    kill_timer.cancel()
                self._inference_lock.release()
                if not self.config.keep_jobs and job_root.exists():
                    shutil.rmtree(job_root, ignore_errors=True)

        return event_stream()


def create_app(
    config_path: str | Path | None = None,
    *,
    initialize_runtime: bool | None = None,
) -> Flask:
    config = DeploymentConfig.load(config_path)
    service = MegaVitService(config)
    should_initialize = (
        config.persistent_runtime
        if initialize_runtime is None else bool(initialize_runtime)
    )
    if should_initialize:
        service.initialize()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes
    app.extensions["megavit_service"] = service

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            profile=config.profile,
            default_question=config.question,
            default_max_new_tokens=config.max_new_tokens,
            max_allowed_new_tokens=config.max_allowed_new_tokens,
        )

    @app.get("/health")
    def health():
        result = service.validate(refresh=True)
        return jsonify(result), 200 if result["ready"] else 503

    @app.get("/v1/models")
    def models():
        return jsonify(
            {
                "profile": config.profile,
                "seq_len": config.seq_len,
                "target_canvases": config.target_canvases,
                "vision_workers": config.vision_workers,
                "outlier_top_k": config.outlier_top_k,
                "question": config.question,
                "system": config.system,
                "max_question_tokens": config.max_question_tokens,
                "max_allowed_new_tokens": config.max_allowed_new_tokens,
                "custom_prompt": True,
                "padding_side": "right",
                "persistent_runtime": config.persistent_runtime,
                "ui": "/",
                "blocking_endpoint": "/v1/video/generate",
                "streaming_endpoint": "/v1/video/generate/stream",
                "streaming_content_type": "application/x-ndjson",
            }
        )

    @app.post("/v1/video/generate")
    def generate():
        uploaded = request.files.get("video")
        if uploaded is None:
            raise DeploymentError("multipart field 'video' is required", 400)
        question = request.form.get("question", config.question)
        try:
            max_new_tokens = int(
                request.form.get("max_new_tokens", config.max_new_tokens)
            )
        except ValueError as error:
            raise DeploymentError("max_new_tokens must be an integer", 422) from error
        return jsonify(
            service.generate(
                uploaded,
                question=question,
                max_new_tokens=max_new_tokens,
            )
        )

    @app.post("/v1/video/generate/stream")
    def generate_stream():
        uploaded = request.files.get("video")
        if uploaded is None:
            raise DeploymentError("multipart field 'video' is required", 400)
        question = request.form.get("question", config.question)
        try:
            max_new_tokens = int(
                request.form.get("max_new_tokens", config.max_new_tokens)
            )
        except ValueError as error:
            raise DeploymentError("max_new_tokens must be an integer", 422) from error
        events = service.stream_generate(
            uploaded,
            question=question,
            max_new_tokens=max_new_tokens,
        )
        response = Response(
            stream_with_context(events),
            mimetype="application/x-ndjson",
        )
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.errorhandler(DeploymentError)
    def deployment_error(error: DeploymentError):
        return jsonify({"error": str(error)}), error.status_code

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error: RequestEntityTooLarge):
        del error
        return jsonify({"error": "video upload exceeds configured limit"}), 413

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    app = create_app(
        args.config,
        initialize_runtime=False if args.validate_only else None,
    )
    service: MegaVitService = app.extensions["megavit_service"]
    if args.validate_only:
        result = service.validate(refresh=True)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["ready"] else 1)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
