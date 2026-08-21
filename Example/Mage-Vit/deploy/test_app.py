#!/usr/bin/env python3
"""Dependency-light Flask deployment contract tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from app import DeploymentConfig, MegaVitService, create_app


class _Upload:
    filename = "sample.mp4"

    @staticmethod
    def save(path: Path) -> None:
        path.write_bytes(b"video")


def _config(root: Path) -> Path:
    path = root / "deploy" / "deployment.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "format": "mage-vl-deployment-v1",
                "profile": "test-s8",
                "model_path": "model",
                "frontend_bin": "build/megavit_frontend",
                "vision_fb": "deploy/models/vision.fb",
                "addon": "deploy/runtime/addon.so",
                "executor_config": "runconfig.json",
                "decoder_dir": "deploy/models/decoder",
                "work_dir": "deploy/var/jobs",
                "threads": 4,
                "vision_workers": 4,
                "max_new_tokens": 8,
                "max_allowed_new_tokens": 16,
                "request_timeout_seconds": 60,
                "max_upload_bytes": 1024,
                "require_npu": False,
                "keep_jobs": False,
                "prefill_bucket": {
                    "seq_len": 8,
                    "target_canvases": 1,
                    "question": "Describe this video.",
                    "system": "You are helpful.",
                    "max_question_tokens": 4,
                    "fb_dir": "deploy/models/prefill",
                },
            }
        )
    )
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        config_path = _config(Path(directory))
        config = DeploymentConfig.load(config_path)
        assert config.seq_len == 8
        assert config.vision_workers == 4
        assert config.qwen_fb_dir == Path(directory) / "deploy/models/prefill"
        service = MegaVitService(config)
        command = service.build_command(
            Path("/tmp/input.mp4"),
            Path("/tmp/output"),
            question=config.question,
            max_new_tokens=8,
        )
        assert "--hardware" in command
        assert command[command.index("--expected-seq-len") + 1] == "8"
        stream_command = service.build_command(
            Path("/tmp/input.mp4"),
            Path("/tmp/output"),
            question="What happened?",
            max_new_tokens=8,
            stream_jsonl=True,
        )
        assert "--stream-jsonl" in stream_command
        service._tokenizer = lambda value, add_special_tokens=False: (  # noqa: ARG005
            SimpleNamespace(input_ids=value.split())
        )
        assert service.question_token_count("one two") == 2
        try:
            service.question_token_count("one two three four five")
        except Exception as error:
            assert "at most 4" in str(error)
        else:
            raise AssertionError("oversized custom question was accepted")
        fake_pipeline = Path(directory) / "fake_pipeline.py"
        fake_pipeline.write_text(
            """import json, sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
prefix = 'MEGAVIT_EVENT '
print(prefix + json.dumps({'type': 'stage_start', 'stage': 'ort-decode'}), flush=True)
print(prefix + json.dumps({'type': 'token', 'token_id': 1, 'token_index': 0, 'text_delta': 'Hi', 'text': 'Hi', 'replace': False}), flush=True)
(out / 'decode.report.json').write_text(json.dumps({'text': 'Hi', 'tokens': [1], 'prompt_tokens': 8, 'generated_tokens': 1, 'ort_tokens_per_second': 9.0, 'session_load_seconds': 0.1, 'decode_loop_seconds': 0.2}))
(out / 'pipeline.report.json').write_text(json.dumps({'stages': []}))
"""
        )
        service.validate = lambda refresh=False: {"ready": True}  # noqa: ARG005
        service.build_command = (  # type: ignore[method-assign]
            lambda video, output, **kwargs: [  # noqa: ARG005
                sys.executable,
                str(fake_pipeline),
                str(output),
            ]
        )
        stream = service.stream_generate(
            _Upload(), question="one two", max_new_tokens=2
        )
        events = [json.loads(line) for line in stream]
        assert [event["type"] for event in events] == [
            "accepted",
            "stage_start",
            "token",
            "done",
        ]
        assert events[2]["time_to_first_token_seconds"] >= 0
        assert events[3]["text"] == "Hi"
        assert events[3]["decode_seconds"] >= 0

        class _PersistentRuntime:
            @staticmethod
            def stream(*args, **kwargs):  # noqa: ARG004
                yield {"type": "stage_start", "stage": "npu-prefill"}
                yield {
                    "type": "token",
                    "token_id": 1,
                    "token_index": 0,
                    "text_delta": "Hi",
                    "text": "Hi",
                    "replace": False,
                }
                yield {
                    "type": "pipeline_done",
                    "total_seconds": 0.3,
                    "stages": [],
                    "decode_report": {
                        "text": "Hi",
                        "tokens": [1],
                        "prompt_tokens": 8,
                        "generated_tokens": 1,
                        "ort_tokens_per_second": 9.0,
                        "session_load_seconds": 0.0,
                        "persistent_session_load_seconds": 5.0,
                        "decode_loop_seconds": 0.2,
                    },
                }

        persistent_service = MegaVitService(config)
        persistent_service._runtime = _PersistentRuntime()
        persistent_service._tokenizer = service._tokenizer
        persistent_service.validate = lambda refresh=False: {  # noqa: ARG005
            "ready": True
        }
        persistent_events = [
            json.loads(line)
            for line in persistent_service.stream_generate(
                _Upload(), question="one two", max_new_tokens=2
            )
        ]
        assert [event["type"] for event in persistent_events] == [
            "accepted",
            "stage_start",
            "token",
            "done",
        ]
        assert persistent_events[-1]["persistent_runtime"] is True
        assert persistent_events[-1][
            "ort_persistent_session_load_seconds"
        ] == 5.0
        app = create_app(config_path)
        client = app.test_client()
        index = client.get("/")
        assert index.status_code == 200
        assert b"Mage-VL Hybrid Demo" in index.data
        assert b"/v1/video/generate/stream" in (
            Path(__file__).parent / "static/app.js"
        ).read_bytes()
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.get_json()["custom_prompt"] is True
        assert models.get_json()["max_question_tokens"] == 4
        assert models.get_json()["vision_workers"] == 4
        health = client.get("/health")
        assert health.status_code == 503
        assert health.get_json()["ready"] is False
        missing = client.post("/v1/video/generate")
        assert missing.status_code == 400
        missing_stream = client.post("/v1/video/generate/stream")
        assert missing_stream.status_code == 400
    print("Mage-Vit Flask deployment tests: OK")


if __name__ == "__main__":
    main()
