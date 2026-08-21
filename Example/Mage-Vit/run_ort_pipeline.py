#!/usr/bin/env python3
"""Run the complete FFmpeg/TFDL-prefill/ONNX Runtime decode pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SDK_ROOT = THIS_DIR.parents[1]
STREAM_PREFIX = "MEGAVIT_EVENT "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video")
    parser.add_argument("--frontend-bin")
    parser.add_argument("--vision-fb")
    parser.add_argument(
        "--target-canvases",
        type=int,
        help=(
            "override the codec frontend canvas count; useful for a fixed "
            "Qwen prefill sequence bucket"
        ),
    )
    parser.add_argument(
        "--expected-seq-len",
        type=int,
        help="reject an assembled prompt that does not match the FB bucket",
    )
    parser.add_argument("--qwen-fb-dir")
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question", default="Describe this video.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--compare-prefill-reference", action="store_true")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument(
        "--addon",
        default=str(SDK_ROOT / "AddonOps/build/libTFDLAddOn.so"),
    )
    parser.add_argument("--executor-config", default=str(THIS_DIR / "runconfig.json"))
    parser.add_argument(
        "--stream-jsonl",
        action="store_true",
        help="forward stage/token progress as prefixed JSON lines",
    )
    return parser.parse_args()


def _emit_stream(enabled: bool, payload: dict[str, object]) -> None:
    if enabled:
        print(STREAM_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _run(
    label: str,
    command: list[str],
    reports: list[dict[str, object]],
    *,
    stream_jsonl: bool = False,
) -> None:
    print(f"[{label}] {' '.join(command)}", flush=True)
    _emit_stream(stream_jsonl, {"type": "stage_start", "stage": label})
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    seconds = time.perf_counter() - started
    reports.append(
        {
            "stage": label,
            "command": command,
            "seconds": seconds,
            "returncode": completed.returncode,
        }
    )
    _emit_stream(
        stream_jsonl,
        {
            "type": "stage_done",
            "stage": label,
            "seconds": seconds,
            "returncode": completed.returncode,
        },
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")


def _require_argument(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for the selected pipeline stages")
    return value


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    frontend = output / "frontend"
    prompt = output / "prompt"
    prefill = output / "prefill"
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    total_started = time.perf_counter()

    if not args.skip_frontend:
        frontend_bin = _require_argument(args.frontend_bin, "--frontend-bin")
        video = _require_argument(args.video, "--video")
        vision_fb = _require_argument(args.vision_fb, "--vision-fb")
        frontend_command = [
            str(Path(frontend_bin)),
            "--video",
            str(Path(video)),
            "--model",
            str(Path(vision_fb)),
            "--addon",
            str(Path(args.addon)),
            "--executor-config",
            str(Path(args.executor_config)),
            "--output-dir",
            str(frontend),
        ]
        if args.target_canvases is not None:
            if args.target_canvases <= 0:
                raise ValueError("--target-canvases must be positive")
            frontend_command.extend(
                ["--target-canvases", str(args.target_canvases)]
            )
        _run(
            "codec-and-vision",
            frontend_command,
            reports,
            stream_jsonl=args.stream_jsonl,
        )
    _require(frontend / "visual_embeddings.f32", "NPU visual embeddings")
    _require(frontend / "manifest.json", "visual frontend manifest")

    if not args.skip_prefill:
        qwen_fb_dir = _require_argument(args.qwen_fb_dir, "--qwen-fb-dir")
        _run(
            "assemble-prompt",
            [
                sys.executable,
                str(THIS_DIR / "Qwen-prefill/prepare_qwen_prefill_prompt.py"),
                "--model-path",
                str(Path(args.model_path)),
                "--bundle",
                str(frontend),
                "--question",
                args.question,
                "--system",
                args.system,
                "--output-dir",
                str(prompt),
                *(
                    ["--pad-to-seq-len", str(args.expected_seq_len)]
                    if args.expected_seq_len is not None
                    else []
                ),
            ],
            reports,
            stream_jsonl=args.stream_jsonl,
        )
        prompt_metadata = json.loads((prompt / "metadata.json").read_text())
        actual_sequence = int(
            prompt_metadata.get("model_seq_len", prompt_metadata["seq_len"])
        )
        if (
            args.expected_seq_len is not None
            and actual_sequence != args.expected_seq_len
        ):
            raise ValueError(
                "assembled prompt sequence length "
                f"{actual_sequence} does not match deployment FB bucket "
                f"{args.expected_seq_len}"
            )
        prefill_command = [
            sys.executable,
            str(THIS_DIR / "Qwen-prefill/run_qwen_prefill_stack.py"),
            "--model-path",
            str(Path(args.model_path)),
            "--fb-dir",
            str(Path(qwen_fb_dir)),
            "--prompt-dir",
            str(prompt),
            "--output-dir",
            str(prefill),
            # Custom-op registration is process-local.  The frontend loading
            # the addon does not register ArmCausalMaskSoftmax in this Python
            # process, and serialized FB addon paths are build-machine paths.
            "--addon-path",
            str(Path(args.addon)),
        ]
        if args.hardware:
            prefill_command.append("--hardware")
        if args.compare_prefill_reference:
            prefill_command.append("--compare-reference")
        _run(
            "npu-prefill",
            prefill_command,
            reports,
            stream_jsonl=args.stream_jsonl,
        )
    _require(prompt / "input_ids.npy", "assembled prompt")
    _require(prefill / "manifest.json", "NPU prefill manifest")
    _require(prefill / "last_token_logits.npy", "NPU prefill logits")

    decode_report = output / "decode.report.json"
    decoder_command = [
        sys.executable,
        str(THIS_DIR / "Qwen-decode-ort/decode_ort.py"),
        "--model-path",
        str(Path(args.model_path)),
        "--decoder-dir",
        str(Path(args.decoder_dir)),
        "--prompt-dir",
        str(prompt),
        "--prefill-dir",
        str(prefill),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--threads",
        str(args.threads),
        "--output-json",
        str(decode_report),
    ]
    if args.stream_jsonl:
        decoder_command.append("--stream-jsonl")
    _run(
        "ort-decode",
        decoder_command,
        reports,
        stream_jsonl=args.stream_jsonl,
    )
    report = {
        "format": "mage-vl-npu-prefill-ort-decode-pipeline-v1",
        "model_path": str(Path(args.model_path)),
        "video": str(Path(args.video)) if args.video else None,
        "question": args.question,
        "target_canvases": args.target_canvases,
        "hardware_prefill": bool(args.hardware),
        "total_seconds": time.perf_counter() - total_started,
        "stages": reports,
        "outputs": {
            "frontend": str(frontend),
            "prompt": str(prompt),
            "prefill": str(prefill),
            "decode_report": str(decode_report),
        },
    }
    (output / "pipeline.report.json").write_text(json.dumps(report, indent=2))
    _emit_stream(
        args.stream_jsonl,
        {
            "type": "pipeline_done",
            "total_seconds": report["total_seconds"],
            "stages": reports,
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
