#!/usr/bin/env python3
"""Process-resident Mage-VL vision, NPU prefill and ORT decode runtime.

The command-line runners remain useful for conversion and diagnostics.  This
module is the deployment path: every TFContext, TFExecutor and ORT session is
constructed once when the Flask worker starts and is reused for all requests.
Only FFmpeg/canvas extraction is delegated to the native frontend process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Generator

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
PYTHON_DIR = PROJECT_ROOT / "python"
PREFILL_DIR = PROJECT_ROOT / "Qwen-prefill"
DECODE_DIR = PROJECT_ROOT / "Qwen-decode-ort"
for directory in (PYTHON_DIR, PREFILL_DIR, DECODE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_mage_vit as mage  # noqa: E402
import decode_ort  # noqa: E402
import export_mage_embeddings as vision_export  # noqa: E402
import ort_qwen_decoder as decoder  # noqa: E402
import qwen3_bridge  # noqa: E402
import qwen_prefill as prefill  # noqa: E402


@dataclass
class PromptBatch:
    hidden: np.ndarray
    position_ids: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def valid_sequence(self) -> int:
        return int(self.metadata["valid_seq_len"])

    def write(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "hidden.npy", self.hidden)
        np.save(output / "position_ids.npy", self.position_ids)
        np.save(output / "input_ids.npy", self.input_ids)
        np.save(output / "attention_mask.npy", self.attention_mask)
        (output / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2)
        )


@dataclass
class PrefillResult:
    keys: list[np.ndarray]
    values: list[np.ndarray]
    logits: np.ndarray
    final_hidden: np.ndarray
    manifest: dict[str, Any]


class PromptAssembler:
    """Tokenizer and embedding table kept resident across requests."""

    def __init__(
        self,
        model_path: Path,
        tokenizer: Any,
        tokenizer_lock: threading.RLock,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer = tokenizer
        self.tokenizer_lock = tokenizer_lock
        self.raw_config = json.loads(
            (self.model_path / "config.json").read_text()
        )
        self.config = prefill.QwenPrefillConfig.from_model(self.model_path)
        self.index = prefill.SafeTensorIndex(self.model_path)
        self.embedding_name = prefill.embedding_weight_name(self.index)
        self.pad_token_id = (
            int(tokenizer.pad_token_id)
            if tokenizer.pad_token_id is not None
            else int(self.raw_config["eos_token_id"])
        )

    def assemble(
        self,
        frontend_root: Path,
        *,
        question: str,
        system: str,
        target_sequence: int,
        fallback_fps: float = 24.0,
    ) -> PromptBatch:
        manifest = json.loads((frontend_root / "manifest.json").read_text())
        vision_content, visual_tokens = qwen3_bridge.build_vision_content(
            manifest, fallback_fps
        )
        visual = np.fromfile(
            frontend_root / "visual_embeddings.f32", dtype=np.float32
        )
        expected_values = visual_tokens * self.config.hidden_size
        if visual.size != expected_values:
            raise ValueError(
                f"visual embedding has {visual.size} values, expected "
                f"{visual_tokens}x{self.config.hidden_size}={expected_values}"
            )
        visual = visual.reshape(visual_tokens, self.config.hidden_size)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": vision_content + "\n" + question,
            },
        ]
        with self.tokenizer_lock:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self.tokenizer(text, return_tensors="np")
        ids = np.asarray(encoded.input_ids, dtype=np.int64)
        mask = np.asarray(encoded.attention_mask, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != mask.shape:
            raise ValueError("tokenizer must return matching [1,S] IDs/mask")
        if not np.all(mask == 1):
            raise ValueError("tokenizer unexpectedly padded the prompt")
        valid_sequence = int(ids.shape[1])
        if valid_sequence > target_sequence:
            raise ValueError(
                f"prompt needs {valid_sequence} tokens, but the selected NPU "
                f"bucket supports at most {target_sequence}"
            )
        input_ids = np.full(
            (1, target_sequence), self.pad_token_id, dtype=np.int64
        )
        attention_mask = np.zeros((1, target_sequence), dtype=np.int64)
        input_ids[:, :valid_sequence] = ids
        attention_mask[:, :valid_sequence] = mask

        unique_ids, inverse = np.unique(input_ids.reshape(-1), return_inverse=True)
        unique_embeddings = self.index.read_rows(
            self.embedding_name, unique_ids
        )
        hidden = unique_embeddings[inverse].reshape(
            1, target_sequence, self.config.hidden_size
        )
        image_token_id = int(self.raw_config["image_token_id"])
        image_mask = (input_ids == image_token_id) & (attention_mask == 1)
        if int(np.count_nonzero(image_mask)) != visual_tokens:
            raise ValueError(
                f"prompt has {int(np.count_nonzero(image_mask))} image tokens, "
                f"frontend has {visual_tokens} visual embeddings"
            )
        hidden[image_mask] = visual
        position_ids = attention_mask.cumsum(axis=-1) - 1
        position_ids[attention_mask == 0] = 1
        metadata = {
            "model_path": str(self.model_path),
            "bundle": str(frontend_root),
            "question": question,
            "system": system,
            "seq_len": target_sequence,
            "model_seq_len": target_sequence,
            "valid_seq_len": valid_sequence,
            "padding_side": "right",
            "pad_token_id": self.pad_token_id,
            "visual_tokens": visual_tokens,
            "hidden_size": self.config.hidden_size,
            "image_token_id": image_token_id,
            "eos_token_id": int(self.raw_config["eos_token_id"]),
            "embedding_weight": self.embedding_name,
            "runtime": "persistent",
            "files": {
                "hidden": "hidden.npy",
                "position_ids": "position_ids.npy",
                "input_ids": "input_ids.npy",
                "attention_mask": "attention_mask.npy",
            },
        }
        return PromptBatch(
            hidden=np.ascontiguousarray(hidden, dtype=np.float32),
            position_ids=np.ascontiguousarray(position_ids, dtype=np.int64),
            input_ids=np.ascontiguousarray(input_ids),
            attention_mask=np.ascontiguousarray(attention_mask),
            metadata=metadata,
        )


class PersistentVisionRuntime:
    def __init__(
        self,
        model_path: Path,
        fb_path: Path,
        executor_config: Path,
        TFContext: Any,
        TFExecutor: Any,
        workers: int = 1,
    ) -> None:
        if workers <= 0:
            raise ValueError("vision workers must be positive")
        config = mage.vision_executor_config(
            True,
            base=json.loads(Path(executor_config).read_text()),
        )
        self.executor_config = config
        started = time.perf_counter()
        self.context = TFContext(path=str(fb_path))
        # The SDK supports one shared-weight context with one executor owned by
        # each inference thread. Never submit two jobs to the same executor.
        self.executors = [
            TFExecutor(self.context, config) for _ in range(int(workers))
        ]
        self.compile_seconds = time.perf_counter() - started
        self.workers = int(workers)
        self.model_path = Path(model_path)
        print(
            f"[persistent-runtime] vision executors={self.workers} ready "
            f"in {self.compile_seconds:.3f}s",
            flush=True,
        )

    @staticmethod
    def _run_partition(
        executor: Any,
        entries: list[tuple[int, dict[str, Any]]],
        frontend_root: Path,
        config: Any,
    ) -> list[tuple[int, np.ndarray, float]]:
        results: list[tuple[int, np.ndarray, float]] = []
        for index, entry in entries:
            raw, _, sin, cos = vision_export.canvas_inputs(
                frontend_root, entry, config
            )
            inputs = executor.GetInputs()
            if len(inputs) != 3:
                raise RuntimeError("vision FB must expose RGB, sin and cos")
            inputs[0].fromNumpy(
                raw.astype(np.float32)
                if "FLOAT" in str(inputs[0].dtype)
                else raw
            )
            inputs[1].fromNumpy(np.ascontiguousarray(sin))
            inputs[2].fromNumpy(np.ascontiguousarray(cos))
            started = time.perf_counter()
            outputs = executor()
            seconds = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError("vision FB must expose one embedding output")
            value = outputs[0].toNumpy().astype(np.float32)
            expected_tokens = len(entry["patch_positions"]) // 4
            if value.size != expected_tokens * config.out_hidden_size:
                raise RuntimeError(
                    f"canvas {index} embedding size {value.size} does not "
                    f"match {expected_tokens}x{config.out_hidden_size}"
                )
            results.append(
                (
                    index,
                    value.reshape(1, expected_tokens, config.out_hidden_size),
                    seconds,
                )
            )
        return results

    def run(self, frontend_root: Path) -> dict[str, Any]:
        manifest = json.loads((frontend_root / "manifest.json").read_text())
        canvas_count = len(manifest["canvases"])
        config = mage.MageVisionConfig.from_model(
            self.model_path,
            (int(manifest["canvas_height"]), int(manifest["canvas_width"])),
        )
        entries = list(enumerate(manifest["canvases"]))
        if not entries:
            raise RuntimeError("frontend manifest contains no canvases")
        active_workers = min(self.workers, len(entries))
        partitions = [entries[index::active_workers] for index in range(active_workers)]
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=active_workers,
            thread_name_prefix="mage-vision",
        ) as pool:
            futures = [
                pool.submit(
                    self._run_partition,
                    self.executors[index],
                    partitions[index],
                    frontend_root,
                    config,
                )
                for index in range(active_workers)
            ]
            indexed = [item for future in futures for item in future.result()]
        wall_seconds = time.perf_counter() - started
        indexed.sort(key=lambda item: item[0])
        embeddings = [item[1] for item in indexed]
        per_canvas = [item[2] for item in indexed]
        combined = np.concatenate(embeddings, axis=1).reshape(-1)
        combined.astype(np.float32).tofile(
            frontend_root / "visual_embeddings.f32"
        )
        return {
            "canvases": canvas_count,
            "visual_tokens": int(combined.size // config.out_hidden_size),
            "execute_seconds": wall_seconds,
            "sum_canvas_execute_seconds": float(sum(per_canvas)),
            "per_canvas_seconds": per_canvas,
            "workers": active_workers,
            "executor_reused": True,
        }


class PersistentPrefillRuntime:
    def __init__(
        self,
        model_path: Path,
        fb_dir: Path,
        seq_len: int,
        TFContext: Any,
        TFExecutor: Any,
    ) -> None:
        self.model_path = Path(model_path)
        self.fb_dir = Path(fb_dir)
        self.seq_len = int(seq_len)
        self.config = prefill.QwenPrefillConfig.from_model(model_path)
        self.checkpoint = prefill.SafeTensorIndex(model_path)
        self.contexts: list[Any] = []
        self.executors: list[Any] = []
        self.compile_seconds: list[float] = []
        self.executor_config = prefill.prefill_executor_config(True)
        for layer in range(self.config.num_hidden_layers):
            artifact = self.fb_dir / f"layer_{layer:02d}_seq_{seq_len}.fb"
            started = time.perf_counter()
            context = TFContext(path=str(artifact))
            executor = TFExecutor(
                context,
                self.executor_config,
            )
            self.compile_seconds.append(time.perf_counter() - started)
            # Both objects intentionally remain alive for the Flask worker.
            self.contexts.append(context)
            self.executors.append(executor)
            print(
                f"[persistent-runtime] prefill layer {layer:02d} ready "
                f"in {self.compile_seconds[-1]:.3f}s",
                flush=True,
            )

    def run(
        self,
        prompt: PromptBatch,
        output: Path,
        *,
        persist_artifacts: bool,
        logits_runner: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> PrefillResult:
        hidden = prompt.hidden
        if hidden.shape[1] != self.seq_len:
            raise ValueError(
                f"prompt width {hidden.shape[1]} does not match S={self.seq_len}"
            )
        valid = prompt.valid_sequence
        positions = prompt.position_ids.reshape(-1)
        sin, cos = prefill.compute_rope(
            positions, self.config.head_dim, self.config.rope_theta
        )
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        layer_reports: list[dict[str, Any]] = []
        total_started = time.perf_counter()
        for layer, executor in enumerate(self.executors):
            started = time.perf_counter()
            hidden_next, key, value = prefill.execute_layer(
                executor, hidden, sin, cos
            )
            seconds = time.perf_counter() - started
            key_valid = np.array(key[:, :, :valid], dtype=np.float16, copy=True)
            value_valid = np.array(
                value[:, :, :valid], dtype=np.float16, copy=True
            )
            keys.append(key_valid)
            values.append(value_valid)
            hidden = np.ascontiguousarray(hidden_next)
            layer_reports.append(
                {
                    "layer": layer,
                    "execute_seconds": seconds,
                    "executor_reused": True,
                }
            )
        final_hidden = np.array(hidden[:, :valid], copy=True)
        logits_started = time.perf_counter()
        if logits_runner is None:
            logits = prefill.compute_final_logits(
                self.model_path,
                final_hidden[:, -1:],
                self.config,
                device="cpu",
                index=self.checkpoint,
            )
            logits_engine = "checkpoint-float"
        else:
            logits = logits_runner(final_hidden[:, -1:])
            logits_engine = "onnxruntime-w8a8"
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        logits_seconds = time.perf_counter() - logits_started
        top10 = np.argsort(logits)[-10:][::-1]
        cache_files: dict[str, dict[str, str]] = {}
        for layer, (key, value) in enumerate(zip(keys, values)):
            key_name = f"layer_{layer:02d}.key.npy"
            value_name = f"layer_{layer:02d}.value.npy"
            cache_files[str(layer)] = {"key": key_name, "value": value_name}
        manifest = {
            "format": decoder.PREFILL_FORMAT,
            "model_path": str(self.model_path),
            "prompt": prompt.metadata,
            "seq_len": valid,
            "valid_seq_len": valid,
            "model_seq_len": self.seq_len,
            "layers": list(range(self.config.num_hidden_layers)),
            "hidden_size": self.config.hidden_size,
            "num_key_value_heads": self.config.num_key_value_heads,
            "head_dim": self.config.head_dim,
            "cache_dtype": "float16",
            "cache_files": cache_files,
            "final_hidden": "final_hidden.npy",
            "logits": {
                "seconds": logits_seconds,
                "engine": logits_engine,
                "top1": int(top10[0]),
                "top10": [int(value) for value in top10],
            },
            "use_hardware": True,
            "executor_lifetime": "process-resident",
            "runtime_reused": True,
            "total_seconds": time.perf_counter() - total_started,
            "layer_reports": layer_reports,
        }
        if persist_artifacts:
            output.mkdir(parents=True, exist_ok=True)
            for layer, (key, value) in enumerate(zip(keys, values)):
                np.save(output / f"layer_{layer:02d}.key.npy", key)
                np.save(output / f"layer_{layer:02d}.value.npy", value)
            np.save(output / "final_hidden.npy", final_hidden)
            np.save(output / "last_token_logits.npy", logits)
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
            (output / "report.json").write_text(
                json.dumps(manifest, indent=2)
            )
        return PrefillResult(
            keys=keys,
            values=values,
            logits=logits,
            final_hidden=final_hidden,
            manifest=manifest,
        )


class InMemoryKvCache:
    def __init__(
        self,
        keys: list[np.ndarray],
        values: list[np.ndarray],
        *,
        length: int,
        capacity: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        if len(keys) != num_layers or len(values) != num_layers:
            raise ValueError("NPU prefill did not return every KV layer")
        self.length = int(length)
        self.capacity = int(capacity)
        expected = (1, num_kv_heads, length, head_dim)
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        for layer, (key, value) in enumerate(zip(keys, values)):
            if key.dtype != np.float16 or value.dtype != np.float16:
                raise TypeError(f"layer {layer} KV must be FP16")
            if key.shape != expected or value.shape != expected:
                raise ValueError(
                    f"layer {layer} KV {key.shape}/{value.shape}, "
                    f"expected {expected}"
                )
            key_store = np.empty(
                (1, capacity, num_kv_heads, head_dim), dtype=np.float16
            )
            value_store = np.empty_like(key_store)
            key_store[:, :length] = key.transpose(0, 2, 1, 3)
            value_store[:, :length] = value.transpose(0, 2, 1, 3)
            self.keys.append(key_store)
            self.values.append(value_store)

    def feeds(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for layer, (key, value) in enumerate(zip(self.keys, self.values)):
            result[decoder.past_key_name(layer)] = key[:, : self.length]
            result[decoder.past_value_name(layer)] = value[:, : self.length]
        return result

    def append(self, outputs: dict[str, np.ndarray]) -> None:
        if self.length >= self.capacity:
            raise ValueError("KV cache capacity exceeded")
        expected = (1, 1, self.keys[0].shape[2], self.keys[0].shape[3])
        for layer, (key_store, value_store) in enumerate(
            zip(self.keys, self.values)
        ):
            key = outputs[decoder.present_key_name(layer)]
            value = outputs[decoder.present_value_name(layer)]
            if key.dtype != np.float16 or value.dtype != np.float16:
                raise TypeError(f"ORT layer {layer} present KV must be FP16")
            if key.shape != expected or value.shape != expected:
                raise ValueError(
                    f"ORT layer {layer} present KV {key.shape}/{value.shape}, "
                    f"expected {expected}"
                )
            key_store[:, self.length : self.length + 1] = key
            value_store[:, self.length : self.length + 1] = value
        self.length += 1


class PersistentOrtDecoder:
    def __init__(
        self,
        model_path: Path,
        decoder_dir: Path,
        *,
        threads: int,
        tokenizer: Any,
        tokenizer_lock: threading.RLock,
    ) -> None:
        self.model_path = Path(model_path)
        self.decoder_dir = Path(decoder_dir)
        self.config = prefill.QwenPrefillConfig.from_model(model_path)
        self.manifest = json.loads(
            (self.decoder_dir / "manifest.json").read_text()
        )
        if self.manifest.get("format") != decoder.DECODER_FORMAT:
            raise ValueError("unsupported ORT decoder manifest")
        decode_ort._validate_checkpoint_fingerprint(
            self.model_path, self.manifest
        )
        self.eos_token = int(self.manifest["eos_token_id"])
        self.tokenizer = tokenizer
        self.tokenizer_lock = tokenizer_lock
        self.embeddings = decode_ort.EmbeddingReader(self.model_path)
        model_file = self.decoder_dir / str(self.manifest["model_file"])
        args = SimpleNamespace(
            threads=int(threads),
            inter_op_threads=1,
            disable_spinning=False,
        )
        self.session, self.session_load_seconds = decode_ort._make_session(
            model_file, args
        )
        decode_ort._validate_session(self.session, self.config)
        final_head_name = self.manifest.get("final_head_model")
        if not isinstance(final_head_name, str) or not final_head_name:
            raise ValueError("decoder manifest does not declare final_head_model")
        self.final_head_model = self.decoder_dir / final_head_name
        self.final_head_session, self.final_head_session_load_seconds = (
            decode_ort._make_session(self.final_head_model, args)
        )
        head_inputs = {item.name for item in self.final_head_session.get_inputs()}
        head_outputs = {item.name for item in self.final_head_session.get_outputs()}
        if head_inputs != {"hidden"} or head_outputs != {"logits"}:
            raise ValueError(
                "final-head ABI mismatch: "
                f"inputs={sorted(head_inputs)} outputs={sorted(head_outputs)}"
            )
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.threads = int(threads)
        self.model_file = model_file
        print(
            "[persistent-runtime] ORT decoder session ready "
            f"in {self.session_load_seconds:.3f}s; W8A8 final head "
            f"in {self.final_head_session_load_seconds:.3f}s",
            flush=True,
        )

    def initial_logits(self, hidden: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(hidden, dtype=np.float32)
        return self.final_head_session.run(["logits"], {"hidden": value})[0]

    def stream(
        self,
        prompt: PromptBatch,
        result: PrefillResult,
        *,
        max_new_tokens: int,
        deadline: float | None,
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        started_at = time.perf_counter()
        valid = prompt.valid_sequence
        capacity = valid + max_new_tokens
        cache_started = time.perf_counter()
        cache = InMemoryKvCache(
            result.keys,
            result.values,
            length=valid,
            capacity=capacity,
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
        )
        cache_seconds = time.perf_counter() - cache_started
        next_token = int(np.asarray(result.logits).reshape(-1).argmax())
        first_position = int(prompt.position_ids[0, valid - 1]) + 1
        generated: list[int] = []
        step_seconds: list[float] = []
        streamed_text = ""

        def token_event(token: int) -> dict[str, Any]:
            nonlocal streamed_text
            generated.append(token)
            with self.tokenizer_lock:
                current_text = self.tokenizer.decode(
                    generated, skip_special_tokens=True
                )
            prefix_preserved = current_text.startswith(streamed_text)
            delta = (
                current_text[len(streamed_text) :]
                if prefix_preserved else current_text
            )
            streamed_text = current_text
            return {
                "type": "token",
                "token_id": token,
                "token_index": len(generated) - 1,
                "text_delta": delta,
                "text": current_text,
                "replace": not prefix_preserved,
                "decoder_elapsed_seconds": time.perf_counter() - started_at,
            }

        if next_token != self.eos_token:
            yield token_event(next_token)
        decode_started = time.perf_counter()
        while generated and len(generated) < max_new_tokens:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("persistent inference request timed out")
            position = first_position + len(generated) - 1
            sin, cos = prefill.compute_rope(
                np.asarray([position], dtype=np.int64),
                self.config.head_dim,
                self.config.rope_theta,
            )
            feeds = cache.feeds()
            feeds[decoder.HIDDEN_INPUT] = self.embeddings.get(next_token)
            feeds[decoder.SIN_INPUT] = sin
            feeds[decoder.COS_INPUT] = cos
            step_started = time.perf_counter()
            values = self.session.run(self.output_names, feeds)
            step_seconds.append(time.perf_counter() - step_started)
            outputs = dict(zip(self.output_names, values))
            cache.append(outputs)
            next_token = int(
                outputs[decoder.LOGITS_OUTPUT].reshape(-1).argmax()
            )
            if next_token == self.eos_token:
                break
            yield token_event(next_token)
        decode_seconds = time.perf_counter() - decode_started
        measured = len(step_seconds)
        report = {
            "format": "mage-qwen3-ort-decode-report-v1",
            "model_path": str(self.model_path),
            "decoder": str(self.model_file),
            "prompt_tokens": valid,
            "model_seq_len": int(prompt.input_ids.shape[1]),
            "generated_tokens": len(generated),
            "npu_seed_token": generated[0] if generated else None,
            "cpu_decode_steps": measured,
            "threads": self.threads,
            "cache_load_seconds": cache_seconds,
            "session_load_seconds": 0.0,
            "persistent_session_load_seconds": self.session_load_seconds,
            "decode_loop_seconds": decode_seconds,
            "ort_step_seconds": step_seconds,
            "ort_mean_step_seconds": float(np.mean(step_seconds))
            if measured else 0.0,
            "ort_p50_step_seconds": float(np.percentile(step_seconds, 50))
            if measured else 0.0,
            "ort_p95_step_seconds": float(np.percentile(step_seconds, 95))
            if measured else 0.0,
            "ort_tokens_per_second": measured / sum(step_seconds)
            if measured else 0.0,
            "tokens": generated,
            "text": streamed_text,
            "runtime_reused": True,
        }
        yield {
            "type": "decoder_done",
            "text": streamed_text,
            "generated_tokens": len(generated),
            "ort_tokens_per_second": report["ort_tokens_per_second"],
            "session_load_seconds": 0.0,
            "persistent_session_load_seconds": self.session_load_seconds,
            "decode_loop_seconds": decode_seconds,
        }
        return report


class PersistentMageRuntime:
    """Own every heavyweight inference object for the worker lifetime."""

    def __init__(self, config: Any) -> None:
        self.config = config
        started = time.perf_counter()
        from TFDL2 import TFContext, TFExecutor
        from TFDL2.utils import LoadCustomOp
        from transformers import AutoTokenizer

        addon_result = LoadCustomOp(str(config.addon))
        self.tokenizer_lock = threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        self.prompt = PromptAssembler(
            config.model_path, self.tokenizer, self.tokenizer_lock
        )
        self.vision = PersistentVisionRuntime(
            config.model_path,
            config.vision_fb,
            config.executor_config,
            TFContext,
            TFExecutor,
            workers=config.vision_workers,
        )
        self.prefill = PersistentPrefillRuntime(
            config.model_path,
            config.qwen_fb_dir,
            config.seq_len,
            TFContext,
            TFExecutor,
        )
        self.decoder = PersistentOrtDecoder(
            config.model_path,
            config.decoder_dir,
            threads=config.threads,
            tokenizer=self.tokenizer,
            tokenizer_lock=self.tokenizer_lock,
        )
        self.startup = {
            "initialized": True,
            "seconds": time.perf_counter() - started,
            "addon_result": addon_result,
            "vision_contexts": 1,
            "vision_executors": len(self.vision.executors),
            "prefill_contexts": len(self.prefill.contexts),
            "prefill_executors": len(self.prefill.executors),
            "vision_compile_seconds": self.vision.compile_seconds,
            "prefill_compile_seconds": float(
                sum(self.prefill.compile_seconds)
            ),
            "vision_executor_config": self.vision.executor_config,
            "prefill_executor_config": self.prefill.executor_config,
            "ort_session_load_seconds": self.decoder.session_load_seconds,
            "ort_final_head_load_seconds": (
                self.decoder.final_head_session_load_seconds
            ),
        }
        print(
            f"[persistent-runtime] ready: vision={len(self.vision.executors)} "
            "prefill=36 ort=1+head "
            f"startup={self.startup['seconds']:.3f}s",
            flush=True,
        )

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("persistent inference request timed out")

    def _codec_frontend(
        self,
        video: Path,
        frontend: Path,
        *,
        timeout: float | None,
        log_path: Path,
    ) -> dict[str, Any]:
        command = [
            str(self.config.frontend_bin),
            "--video",
            str(video),
            "--output-dir",
            str(frontend),
            "--target-canvases",
            str(self.config.target_canvases),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write("[codec-frontend] " + " ".join(command) + "\n")
            log.write(completed.stdout)
            log.write(completed.stderr)
        if completed.returncode:
            raise RuntimeError(
                "codec frontend failed with return code "
                f"{completed.returncode}: "
                f"{(completed.stderr or completed.stdout)[-4000:]}"
            )
        return {"command": command, "returncode": completed.returncode}

    def stream(
        self,
        video: Path,
        output: Path,
        *,
        question: str,
        system: str,
        max_new_tokens: int,
        timeout_seconds: int,
        log_path: Path,
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        output.mkdir(parents=True, exist_ok=True)
        frontend = output / "frontend"
        prompt_root = output / "prompt"
        prefill_root = output / "prefill"
        decode_path = output / "decode.report.json"
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        total_started = time.perf_counter()
        stages: list[dict[str, Any]] = []

        yield {"type": "stage_start", "stage": "codec-and-vision"}
        started = time.perf_counter()
        remaining = (
            max(0.001, deadline - time.monotonic())
            if deadline is not None else None
        )
        codec = self._codec_frontend(
            video, frontend, timeout=remaining, log_path=log_path
        )
        vision = self.vision.run(frontend)
        seconds = time.perf_counter() - started
        stages.append(
            {
                "stage": "codec-and-vision",
                "seconds": seconds,
                "returncode": 0,
                "codec": codec,
                "vision": vision,
                "runtime_reused": True,
            }
        )
        yield {
            "type": "stage_done",
            "stage": "codec-and-vision",
            "seconds": seconds,
            "returncode": 0,
        }
        self._check_deadline(deadline)

        yield {"type": "stage_start", "stage": "assemble-prompt"}
        started = time.perf_counter()
        prompt = self.prompt.assemble(
            frontend,
            question=question,
            system=system,
            target_sequence=self.config.seq_len,
        )
        if self.config.keep_jobs:
            prompt.write(prompt_root)
        seconds = time.perf_counter() - started
        stages.append(
            {
                "stage": "assemble-prompt",
                "seconds": seconds,
                "returncode": 0,
                "runtime_reused": True,
            }
        )
        yield {
            "type": "stage_done",
            "stage": "assemble-prompt",
            "seconds": seconds,
            "returncode": 0,
        }
        self._check_deadline(deadline)

        yield {"type": "stage_start", "stage": "npu-prefill"}
        started = time.perf_counter()
        prefill_result = self.prefill.run(
            prompt,
            prefill_root,
            persist_artifacts=self.config.keep_jobs,
            logits_runner=self.decoder.initial_logits,
        )
        seconds = time.perf_counter() - started
        stages.append(
            {
                "stage": "npu-prefill",
                "seconds": seconds,
                "returncode": 0,
                "runtime_reused": True,
            }
        )
        yield {
            "type": "stage_done",
            "stage": "npu-prefill",
            "seconds": seconds,
            "returncode": 0,
        }
        self._check_deadline(deadline)

        yield {"type": "stage_start", "stage": "ort-decode"}
        started = time.perf_counter()
        decoder_stream = self.decoder.stream(
            prompt,
            prefill_result,
            max_new_tokens=max_new_tokens,
            deadline=deadline,
        )
        while True:
            try:
                event = next(decoder_stream)
            except StopIteration as completed:
                decode_report = completed.value
                break
            else:
                yield event
        seconds = time.perf_counter() - started
        stages.append(
            {
                "stage": "ort-decode",
                "seconds": seconds,
                "returncode": 0,
                "runtime_reused": True,
            }
        )
        yield {
            "type": "stage_done",
            "stage": "ort-decode",
            "seconds": seconds,
            "returncode": 0,
        }
        if self.config.keep_jobs:
            decode_path.write_text(json.dumps(decode_report, indent=2))
        pipeline = {
            "format": "mage-vl-persistent-runtime-pipeline-v1",
            "profile": self.config.profile,
            "model_path": str(self.config.model_path),
            "video": str(video),
            "question": question,
            "target_canvases": self.config.target_canvases,
            "vision_workers": self.config.vision_workers,
            "prefill_seq_len": self.config.seq_len,
            "outlier_top_k": self.config.outlier_top_k,
            "hardware_prefill": True,
            "persistent_runtime": True,
            "startup": self.startup,
            "total_seconds": time.perf_counter() - total_started,
            "stages": stages,
            "outputs": {
                "frontend": str(frontend),
                "prompt": str(prompt_root),
                "prefill": str(prefill_root),
                "decode_report": str(decode_path),
            },
        }
        if self.config.keep_jobs:
            (output / "pipeline.report.json").write_text(
                json.dumps(pipeline, indent=2)
            )
        final_event = {
            "type": "pipeline_done",
            "total_seconds": pipeline["total_seconds"],
            "stages": stages,
            "decode_report": decode_report,
        }
        yield final_event
        return final_event
