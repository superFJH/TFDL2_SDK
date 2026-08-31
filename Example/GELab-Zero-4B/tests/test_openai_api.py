#!/usr/bin/env python3
"""Dependency-light contract tests for the OpenAI-compatible HTTP schema."""

from __future__ import annotations

import base64
import sys
import tempfile
import threading
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "deploy"))
from api_server import (  # noqa: E402
    GelabService,
    handler_factory,
    openai_completion_chunk,
    openai_completion_response,
    parse_openai_chat_request,
)
from http.server import ThreadingHTTPServer  # noqa: E402


class _FakeService:
    openai_model_id = "gelab-zero-4b"
    max_request_bytes = 1024 * 1024
    max_new_tokens = 1024

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def openai_models(self) -> dict[str, object]:
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

    def chat_completion_from_request(
        self,
        request: dict[str, object],
        *,
        on_decode_token: object = None,
        completion_id: str | None = None,
        created: int | None = None,
    ) -> dict[str, object]:
        if on_decode_token is not None:
            on_decode_token("Tap ")
            on_decode_token("60 mins.")
        return openai_completion_response(
            request,
            {
                "result": {
                    "text": "Tap 60 mins.",
                    "generated_tokens": 4,
                    "prompt_tokens": 174,
                }
            },
            completion_id or "chatcmpl-fake",
            created if created is not None else 1,
        )


def main() -> None:
    request = parse_openai_chat_request(
        {
            "model": "gelab-zero-4b",
            "messages": [
                {"role": "system", "content": "You are a precise GUI agent."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "file:///mnt/ui/screen%201.png"},
                        },
                        {"type": "text", "text": "tap the 60 mins option"},
                    ],
                },
            ],
            "max_tokens": 32,
        },
        "gelab-zero-4b",
    )
    assert request["image_path"] == "/mnt/ui/screen 1.png"
    assert request["system"] == "You are a precise GUI agent."
    assert request["question"] == "User: tap the 60 mins option"
    assert request["max_new_tokens"] == 32
    assert request["stream"] is False

    data_url = "data:image/png;base64," + base64.b64encode(b"png-test-bytes").decode()
    data_request = parse_openai_chat_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "describe this"},
                    ],
                }
            ]
        },
        "gelab-zero-4b",
    )
    assert data_request["image_path"] == data_url
    bare_service = object.__new__(GelabService)
    bare_service.remote_media_max_bytes = 1024
    with tempfile.TemporaryDirectory() as temporary:
        media, report = bare_service._materialize_media(
            data_url, "image", Path(temporary)
        )
        assert media.read_bytes() == b"png-test-bytes"
        assert report is not None and report["source"] == "data-url"

    pipeline = {
        "result": {
            "text": "Tap 60 mins.",
            "generated_tokens": 4,
            "prompt_tokens": 174,
        }
    }
    response = openai_completion_response(request, pipeline, "chatcmpl-test", 1)
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Tap 60 mins.",
    }
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"] == {
        "prompt_tokens": 174,
        "completion_tokens": 4,
        "total_tokens": 178,
    }
    chunk = openai_completion_chunk(
        completion_id="chatcmpl-test",
        created=1,
        model="gelab-zero-4b",
        delta={"content": "Tap "},
        finish_reason=None,
    )
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"]["content"] == "Tap "

    try:
        parse_openai_chat_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "/a.png"}},
                            {"type": "image_url", "image_url": {"url": "/b.png"}},
                            {"type": "text", "text": "compare"},
                        ],
                    }
                ]
            },
            "gelab-zero-4b",
        )
    except ValueError as error:
        assert "exactly one image_url" in str(error)
    else:
        raise AssertionError("multiple images must be rejected")

    fake = _FakeService()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        direct_open = build_opener(ProxyHandler({})).open
        with direct_open(base_url + "/v1/models") as http_response:
            models = __import__("json").loads(http_response.read())
        assert models["object"] == "list"
        assert models["data"][0]["id"] == "gelab-zero-4b"

        body = __import__("json").dumps(
            {
                "model": "gelab-zero-4b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "/mnt/ui.png"}},
                            {"type": "text", "text": "set focus time to 60 mins"},
                        ],
                    }
                ],
            }
        ).encode()
        http_request = Request(
            base_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with direct_open(http_request) as http_response:
            completion = __import__("json").loads(http_response.read())
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["message"]["content"] == "Tap 60 mins."

        stream_body = __import__("json").dumps(
            {
                "model": "gelab-zero-4b",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "/mnt/ui.png"}},
                            {"type": "text", "text": "set focus time to 60 mins"},
                        ],
                    }
                ],
            }
        ).encode()
        stream_request = Request(
            base_url + "/v1/chat/completions",
            data=stream_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with direct_open(stream_request) as http_response:
            stream_text = http_response.read().decode()
        assert "chat.completion.chunk" in stream_text
        assert '"content": "Tap "' in stream_text
        assert '"content": "60 mins."' in stream_text
        assert stream_text.endswith("data: [DONE]\n\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("GELab OpenAI API schema tests: OK")


if __name__ == "__main__":
    main()
