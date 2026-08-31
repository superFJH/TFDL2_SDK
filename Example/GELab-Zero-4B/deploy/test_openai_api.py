#!/usr/bin/env python3
"""Test GELab's OpenAI-compatible endpoint with all Android UI holdout images.

Each image is sent as a base64 data URL, so this validates the client/server
OpenAI boundary without requiring the test images to exist on the NPU host.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "calibration/screenspot-android-holdout/images"
DEFAULT_MANIFEST = PROJECT_ROOT / "calibration/screenspot-android-holdout/manifest.json"
DEFAULT_PROMPTS = (
    ("dataset_instruction", "{instruction}"),
    ("screen_description", "Describe this mobile UI briefly and accurately."),
    ("next_action", "What concrete UI target should the user tap or interact with next?"),
    ("controls", "Identify the main visible controls, text, and their current states."),
)


@dataclass(frozen=True)
class ImageCase:
    path: Path
    instruction: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://10.10.13.166:8000/v1",
        help="OpenAI-compatible base URL; /v1 is appended if omitted",
    )
    parser.add_argument("--model", default="gelab-zero-4b")
    parser.add_argument("--api-key", default="local", help="sent as Bearer auth")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="custom question; repeat to replace the default four-question suite",
    )
    parser.add_argument(
        "--mode",
        choices=("stream", "non-stream", "both"),
        default="stream",
        help="response format to verify for every image/question pair",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--max-cases",
        type=int,
        help="limit image count for a quick smoke test; default tests every image",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="default: deploy/var/reports/openai-api-<timestamp>.json",
    )
    return parser.parse_args()


def normalize_base_url(url: str) -> str:
    base = url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def load_cases(images_dir: Path, manifest_path: Path) -> list[ImageCase]:
    images_dir = images_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text())
        records = document.get("records")
        if not isinstance(records, list):
            raise ValueError(f"manifest records must be a list: {manifest_path}")
        cases: list[ImageCase] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("manifest contains a non-object record")
            relative = record.get("image")
            if not isinstance(relative, str):
                raise ValueError("manifest record is missing image")
            candidate = (manifest_path.parent / relative).resolve()
            if not candidate.is_file():
                candidate = images_dir / Path(relative).name
            if not candidate.is_file():
                raise FileNotFoundError(f"manifest image is unavailable: {relative}")
            cases.append(
                ImageCase(
                    path=candidate,
                    instruction=str(record.get("instruction") or ""),
                    source="manifest",
                )
            )
        if cases:
            return cases
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory is unavailable: {images_dir}")
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    cases = [
        ImageCase(path, "", "images-dir")
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not cases:
        raise FileNotFoundError(f"no image files found in {images_dir}")
    return cases


def prompt_suite(case: ImageCase, custom_questions: list[str]) -> list[tuple[str, str]]:
    if custom_questions:
        return [
            (f"custom_{index:02d}", question)
            for index, question in enumerate(custom_questions, start=1)
        ]
    return [
        (label, template.format(instruction=case.instruction))
        for label, template in DEFAULT_PROMPTS
        if label != "dataset_instruction" or case.instruction
    ]


def image_data_url(image_path: Path) -> tuple[str, int, str]:
    raw = image_path.read_bytes()
    content_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return (
        "data:" + content_type + ";base64," + base64.b64encode(raw).decode("ascii"),
        len(raw),
        content_type,
    )


def chat_payload(
    *, model: str, image_url: str, question: str, max_tokens: int, stream: bool
) -> dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "messages": [
            {"role": "system", "content": "You are a helpful GUI agent."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            },
        ],
        "max_tokens": max_tokens,
    }


def open_json(
    opener: Any,
    url: str,
    body: dict[str, Any] | None,
    api_key: str,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers = {"Authorization": "Bearer " + api_key}
    encoded = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=encoded, headers=headers, method="POST" if body else "GET")
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read()
            return (
                int(response.status),
                json.loads(raw.decode("utf-8")),
                {key.lower(): value for key, value in response.headers.items()},
            )
    except HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"request failed: {error}") from error


def open_stream(
    opener: Any,
    url: str,
    body: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    chunks: list[str] = []
    done = False
    event_count = 0
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/event-stream" not in content_type:
                raise RuntimeError(f"expected text/event-stream, got {content_type}")
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if data == "[DONE]":
                    done = True
                    break
                event_count += 1
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError("stream error: " + json.dumps(event["error"]))
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("SSE event is missing choices")
                delta = choices[0].get("delta", {})
                if not isinstance(delta, dict):
                    raise RuntimeError("SSE choice delta must be an object")
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise RuntimeError("SSE delta content must be a string")
                    chunks.append(content)
    except HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"request failed: {error}") from error
    if not done:
        raise RuntimeError("SSE stream ended without [DONE]")
    if event_count < 2:
        raise RuntimeError("SSE stream did not contain role and terminal events")
    return "".join(chunks), {"sse_events": event_count, "done": done}


def run_one(
    *,
    opener: Any,
    base_url: str,
    api_key: str,
    model: str,
    case: ImageCase,
    prompt_label: str,
    question: str,
    max_tokens: int,
    stream: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    image_url, image_bytes, content_type = image_data_url(case.path)
    payload = chat_payload(
        model=model,
        image_url=image_url,
        question=question,
        max_tokens=max_tokens,
        stream=stream,
    )
    started = time.perf_counter()
    record: dict[str, Any] = {
        "image": str(case.path),
        "image_bytes": image_bytes,
        "image_content_type": content_type,
        "image_transport": "data-url-base64",
        "source": case.source,
        "dataset_instruction": case.instruction or None,
        "prompt_label": prompt_label,
        "question": question,
        "stream": stream,
    }
    if stream:
        text, stream_info = open_stream(
            opener, base_url + "/chat/completions", payload, api_key, timeout_seconds
        )
        record.update(stream_info)
        record["text"] = text
    else:
        status, response, _headers = open_json(
            opener, base_url + "/chat/completions", payload, api_key, timeout_seconds
        )
        if status != 200 or response.get("object") != "chat.completion":
            raise RuntimeError("non-stream response is not an OpenAI chat completion")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("non-stream response is missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("non-stream response is missing assistant message content")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("non-stream response is missing usage")
        record["text"] = message["content"]
        record["response_id"] = response.get("id")
        record["usage"] = usage
        record["finish_reason"] = choices[0].get("finish_reason")
    record["duration_seconds"] = time.perf_counter() - started
    return record


def checkpoint_report(report: dict[str, Any], output: Path, *, completed: bool) -> None:
    """Atomically preserve every completed remote request for long evaluations."""
    report["successes"] = len(report["results"])
    report["failures_count"] = len(report["failures"])
    report["processed_requests"] = report["successes"] + report["failures_count"]
    report["completed"] = completed
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0 or args.timeout_seconds <= 0:
        raise ValueError("--max-tokens and --timeout-seconds must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive when specified")

    base_url = normalize_base_url(args.base_url)
    cases = load_cases(args.images_dir, args.manifest)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    modes = (
        (True,)
        if args.mode == "stream"
        else (False,)
        if args.mode == "non-stream"
        else (False, True)
    )
    total = sum(len(prompt_suite(case, args.question)) for case in cases) * len(modes)
    opener = build_opener(ProxyHandler({}))
    started = time.perf_counter()
    output = args.output_json or (
        DEPLOY_DIR / "var/reports" / ("openai-api-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "gelab-openai-api-eval-v1",
        "base_url": base_url,
        "model": args.model,
        "mode": args.mode,
        "max_tokens": args.max_tokens,
        "images": len(cases),
        "planned_requests": total,
        "results": [],
        "failures": [],
    }

    try:
        status, models, _headers = open_json(
            opener, base_url + "/models", None, args.api_key, args.timeout_seconds
        )
        if status != 200 or models.get("object") != "list":
            raise RuntimeError("GET /models is not an OpenAI model-list response")
        report["models"] = models
    except Exception as error:
        report["models_error"] = str(error)
        print("[models] FAIL " + str(error), file=sys.stderr, flush=True)

    index = 0
    for case in cases:
        for prompt_label, question in prompt_suite(case, args.question):
            for stream in modes:
                index += 1
                mode_label = "stream" if stream else "non-stream"
                try:
                    result = run_one(
                        opener=opener,
                        base_url=base_url,
                        api_key=args.api_key,
                        model=args.model,
                        case=case,
                        prompt_label=prompt_label,
                        question=question,
                        max_tokens=args.max_tokens,
                        stream=stream,
                        timeout_seconds=args.timeout_seconds,
                    )
                    report["results"].append(result)
                    preview = result["text"].replace("\n", " ")[:100]
                    print(
                        f"[{index:03d}/{total:03d}] OK {case.path.name} "
                        f"{prompt_label}/{mode_label} {result['duration_seconds']:.2f}s {preview}",
                        flush=True,
                    )
                    checkpoint_report(report, output, completed=False)
                except Exception as error:
                    failure = {
                        "image": str(case.path),
                        "prompt_label": prompt_label,
                        "question": question,
                        "stream": stream,
                        "error": str(error),
                    }
                    report["failures"].append(failure)
                    print(
                        f"[{index:03d}/{total:03d}] FAIL {case.path.name} "
                        f"{prompt_label}/{mode_label}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    checkpoint_report(report, output, completed=False)

    report["duration_seconds"] = time.perf_counter() - started
    checkpoint_report(report, output, completed=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "successes": report["successes"],
                "failures": report["failures_count"],
                "duration_seconds": report["duration_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if report.get("models_error") or report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
