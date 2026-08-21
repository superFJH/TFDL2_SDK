#!/usr/bin/env python3
"""Continue greedy decode in ONNX Runtime from a TFDL/NPU prefill bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import ort_qwen_decoder as decoder


STREAM_PREFIX = "MEGAVIT_EVENT "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--prefill-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--disable-spinning", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--stream-jsonl",
        action="store_true",
        help="emit token progress as prefixed JSON lines on stdout",
    )
    return parser.parse_args()


def _emit_stream(enabled: bool, payload: dict[str, object]) -> None:
    if enabled:
        print(STREAM_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _validate_checkpoint_fingerprint(
    model_root: Path, manifest: dict[str, object]
) -> None:
    expected = manifest.get("checkpoint_fingerprint", {})
    paths = {
        "config_sha256": model_root / "config.json",
        "weight_index_sha256": model_root / "model.safetensors.index.json",
    }
    for name, path in paths.items():
        if name not in expected:
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[name]:
            raise ValueError(f"decoder checkpoint fingerprint mismatch for {path.name}")


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(item) for item in getattr(value, "shape"))


class ExternalKvCache:
    """Preallocated token-major FP16 cache shared with ORT without copies."""

    def __init__(
        self,
        prefill_root: Path,
        manifest: dict[str, object],
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity: int,
    ) -> None:
        self.length = int(manifest["seq_len"])
        if capacity < self.length:
            raise ValueError("KV capacity is smaller than the prefill sequence")
        self.capacity = int(capacity)
        self.keys: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        layers = [int(value) for value in manifest["layers"]]
        if layers != list(range(num_layers)):
            raise ValueError("ORT decode requires a complete ordered NPU KV bundle")
        files = manifest["cache_files"]
        expected_npu = (1, num_kv_heads, self.length, head_dim)
        for layer in layers:
            item = files[str(layer)]
            key = np.load(prefill_root / item["key"], mmap_mode="r")
            value = np.load(prefill_root / item["value"], mmap_mode="r")
            if key.dtype != np.float16 or value.dtype != np.float16:
                raise TypeError(
                    f"layer {layer} KV must be FP16, got {key.dtype}/{value.dtype}"
                )
            if _shape(key) != expected_npu or _shape(value) != expected_npu:
                raise ValueError(
                    f"layer {layer} KV shape {_shape(key)}/{_shape(value)}, "
                    f"expected {expected_npu}"
                )
            key_store = np.empty(
                (1, capacity, num_kv_heads, head_dim), dtype=np.float16
            )
            value_store = np.empty_like(key_store)
            key_store[:, : self.length] = key.transpose(0, 2, 1, 3)
            value_store[:, : self.length] = value.transpose(0, 2, 1, 3)
            self.keys.append(key_store)
            self.values.append(value_store)

    def feeds(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for layer, (key, value) in enumerate(zip(self.keys, self.values)):
            key_view = key[:, : self.length]
            value_view = value[:, : self.length]
            if not key_view.flags.c_contiguous or not value_view.flags.c_contiguous:
                raise RuntimeError(
                    "token-major KV prefix unexpectedly became non-contiguous"
                )
            result[decoder.past_key_name(layer)] = key_view
            result[decoder.past_value_name(layer)] = value_view
        return result

    def append(self, outputs: dict[str, np.ndarray]) -> None:
        if self.length >= self.capacity:
            raise ValueError("KV cache capacity exceeded")
        expected = (1, 1, self.keys[0].shape[2], self.keys[0].shape[3])
        for layer, (key_store, value_store) in enumerate(zip(self.keys, self.values)):
            key = outputs[decoder.present_key_name(layer)]
            value = outputs[decoder.present_value_name(layer)]
            if key.dtype != np.float16 or value.dtype != np.float16:
                raise TypeError(
                    f"ORT layer {layer} present KV is not FP16: "
                    f"{key.dtype}/{value.dtype}"
                )
            if key.shape != expected or value.shape != expected:
                raise ValueError(
                    f"ORT layer {layer} present shape {key.shape}/{value.shape}, "
                    f"expected {expected}"
                )
            key_store[:, self.length : self.length + 1] = key
            value_store[:, self.length : self.length + 1] = value
        self.length += 1


class EmbeddingReader:
    def __init__(self, model_path: Path) -> None:
        self.index = decoder.prefill.SafeTensorIndex(model_path)
        self.name = decoder.prefill.embedding_weight_name(self.index)
        self.cache: dict[int, np.ndarray] = {}

    def get(self, token: int) -> np.ndarray:
        if token not in self.cache:
            value = self.index.read_rows(self.name, np.asarray([token], dtype=np.int64))
            self.cache[token] = np.ascontiguousarray(
                value.reshape(1, 1, -1), dtype=np.float32
            )
        return self.cache[token]


def _make_session(model_path: Path, args: argparse.Namespace):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(args.threads)
    options.inter_op_num_threads = int(args.inter_op_threads)
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.log_severity_level = 2
    options.add_session_config_entry(
        "session.intra_op.allow_spinning",
        "0" if args.disable_spinning else "1",
    )
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session, time.perf_counter() - started


def _validate_session(session, config: decoder.prefill.QwenPrefillConfig) -> None:
    inputs = {item.name for item in session.get_inputs()}
    outputs = {item.name for item in session.get_outputs()}
    expected_inputs = decoder.expected_input_names(config.num_hidden_layers)
    expected_outputs = decoder.expected_output_names(config.num_hidden_layers)
    if inputs != expected_inputs:
        raise ValueError(
            f"decoder input ABI mismatch: missing={sorted(expected_inputs - inputs)}, "
            f"extra={sorted(inputs - expected_inputs)}"
        )
    if outputs != expected_outputs:
        raise ValueError(
            f"decoder output ABI mismatch: missing={sorted(expected_outputs - outputs)}, "
            f"extra={sorted(outputs - expected_outputs)}"
        )


def main() -> None:
    process_started = time.perf_counter()
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.threads < 0 or args.inter_op_threads <= 0:
        raise ValueError("thread counts must be non-negative/positive")
    model_root = Path(args.model_path)
    decoder_root = Path(args.decoder_dir)
    prompt_root = Path(args.prompt_dir)
    prefill_root = Path(args.prefill_dir)
    config = decoder.prefill.QwenPrefillConfig.from_model(model_root)
    decoder_manifest = _read_json(decoder_root / "manifest.json")
    prefill_manifest = _read_json(prefill_root / "manifest.json")
    if decoder_manifest.get("format") != decoder.DECODER_FORMAT:
        raise ValueError("unsupported ORT decoder manifest")
    if prefill_manifest.get("format") != decoder.PREFILL_FORMAT:
        raise ValueError("unsupported NPU prefill manifest")
    _validate_checkpoint_fingerprint(model_root, decoder_manifest)
    manifest_config = decoder_manifest.get("config", {})
    for name in (
        "hidden_size",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    ):
        if int(manifest_config[name]) != int(getattr(config, name)):
            raise ValueError(f"decoder/model config mismatch at {name}")

    input_ids = np.load(prompt_root / "input_ids.npy")
    attention_mask = np.load(prompt_root / "attention_mask.npy")
    position_ids = np.load(prompt_root / "position_ids.npy")
    if input_ids.shape != attention_mask.shape or input_ids.shape != position_ids.shape:
        raise ValueError("prompt IDs, mask and position IDs must have identical shapes")
    if input_ids.shape[0] != 1:
        raise ValueError("the ORT decoder supports batch size one")
    model_sequence = int(input_ids.shape[1])
    prompt_metadata_path = prompt_root / "metadata.json"
    prompt_metadata = (
        _read_json(prompt_metadata_path) if prompt_metadata_path.exists() else {}
    )
    prompt_sequence = int(
        prompt_metadata.get("valid_seq_len", int(attention_mask.sum()))
    )
    if not 0 < prompt_sequence <= model_sequence:
        raise ValueError("invalid prompt valid_seq_len")
    expected_mask = np.zeros_like(attention_mask)
    expected_mask[:, :prompt_sequence] = 1
    if not np.array_equal(attention_mask, expected_mask):
        raise ValueError("prompt must use right-padding after one valid prefix")
    prefill_sequence = int(
        prefill_manifest.get("valid_seq_len", prefill_manifest["seq_len"])
    )
    if prefill_sequence != prompt_sequence:
        raise ValueError("prompt and NPU prefill valid sequence lengths differ")
    if int(prefill_manifest.get("model_seq_len", model_sequence)) != model_sequence:
        raise ValueError("prompt and NPU prefill model sequence lengths differ")

    capacity = prompt_sequence + args.max_new_tokens
    cache_started = time.perf_counter()
    cache = ExternalKvCache(
        prefill_root,
        prefill_manifest,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        capacity=capacity,
    )
    cache_load_seconds = time.perf_counter() - cache_started
    embeddings = EmbeddingReader(model_root)
    logits = np.load(prefill_root / "last_token_logits.npy")
    if logits.size != config.vocab_size:
        raise ValueError(
            f"prefill logits contain {logits.size} values, expected {config.vocab_size}"
        )
    next_token = int(np.asarray(logits).reshape(-1).argmax())

    model_file = decoder_root / str(decoder_manifest["model_file"])
    eos_token = int(decoder_manifest["eos_token_id"])
    first_position = int(position_ids[0, prompt_sequence - 1]) + 1
    generated: list[int] = []
    step_seconds: list[float] = []
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)
    streamed_text = ""

    def append_token(token: int) -> None:
        nonlocal streamed_text
        generated.append(token)
        current_text = tokenizer.decode(generated, skip_special_tokens=True)
        prefix_preserved = current_text.startswith(streamed_text)
        delta = (
            current_text[len(streamed_text) :]
            if prefix_preserved
            else current_text
        )
        streamed_text = current_text
        _emit_stream(
            args.stream_jsonl,
            {
                "type": "token",
                "token_id": token,
                "token_index": len(generated) - 1,
                "text_delta": delta,
                "text": current_text,
                "replace": not prefix_preserved,
                "decoder_elapsed_seconds": time.perf_counter()
                - process_started,
            },
        )

    # The NPU prefill logits already contain the seed token. Emit it before
    # loading the large ORT session so session initialization is not charged
    # to user-visible time-to-first-token.
    if next_token != eos_token:
        append_token(next_token)

    session_load_seconds = 0.0
    session = None
    output_names: list[str] = []
    if generated and len(generated) < args.max_new_tokens:
        session, session_load_seconds = _make_session(model_file, args)
        _validate_session(session, config)
        output_names = [item.name for item in session.get_outputs()]

    decode_started = time.perf_counter()
    while generated and len(generated) < args.max_new_tokens:
        position = first_position + len(generated) - 1
        sin, cos = decoder.prefill.compute_rope(
            np.asarray([position], dtype=np.int64),
            config.head_dim,
            config.rope_theta,
        )
        feeds = cache.feeds()
        feeds[decoder.HIDDEN_INPUT] = embeddings.get(next_token)
        feeds[decoder.SIN_INPUT] = sin
        feeds[decoder.COS_INPUT] = cos
        started = time.perf_counter()
        assert session is not None
        values = session.run(output_names, feeds)
        step_seconds.append(time.perf_counter() - started)
        outputs = dict(zip(output_names, values))
        cache.append(outputs)
        next_token = int(outputs[decoder.LOGITS_OUTPUT].reshape(-1).argmax())
        if next_token == eos_token:
            break
        append_token(next_token)

    decode_seconds = time.perf_counter() - decode_started
    text = streamed_text
    measured = len(step_seconds)
    report = {
        "format": "mage-qwen3-ort-decode-report-v1",
        "model_path": str(model_root),
        "decoder": str(model_file),
        "prompt_dir": str(prompt_root),
        "prefill_dir": str(prefill_root),
        "prompt_tokens": prompt_sequence,
        "model_seq_len": model_sequence,
        "generated_tokens": len(generated),
        "npu_seed_token": generated[0] if generated else None,
        "cpu_decode_steps": measured,
        "threads": args.threads,
        "cache_load_seconds": cache_load_seconds,
        "session_load_seconds": session_load_seconds,
        "decode_loop_seconds": decode_seconds,
        "ort_step_seconds": step_seconds,
        "ort_mean_step_seconds": float(np.mean(step_seconds)) if measured else 0.0,
        "ort_p50_step_seconds": float(np.percentile(step_seconds, 50))
        if measured
        else 0.0,
        "ort_p95_step_seconds": float(np.percentile(step_seconds, 95))
        if measured
        else 0.0,
        "ort_tokens_per_second": measured / sum(step_seconds) if measured else 0.0,
        "tokens": generated,
        "text": text,
    }
    print(text)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"tokens", "text", "ort_step_seconds"}
            },
            indent=2,
        )
    )
    report_path = (
        Path(args.output_json)
        if args.output_json
        else (prefill_root / "ort_decode.report.json")
    )
    report_path.write_text(json.dumps(report, indent=2))
    _emit_stream(
        args.stream_jsonl,
        {
            "type": "decoder_done",
            "text": text,
            "generated_tokens": len(generated),
            "ort_tokens_per_second": report["ort_tokens_per_second"],
            "session_load_seconds": session_load_seconds,
            "decode_loop_seconds": decode_seconds,
        },
    )


if __name__ == "__main__":
    main()
