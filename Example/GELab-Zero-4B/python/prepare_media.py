#!/usr/bin/env python3
"""Prepare one image/video with the official Qwen3-VL processor contract.

The deployment uses a fixed spatial bucket. Input media is letterboxed before
the official normalize/patchify code runs; no visual tensor layout is
reimplemented here. Video sampling follows Qwen3VLVideoProcessor's uniform
FPS rule, while OpenCV supplies timestamps without a PyAV dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint import ModelContract


def _letterbox_rgb(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"expected RGB HWC frame, got {source.shape}")
    scale = min(width / source.shape[1], height / source.shape[0])
    resized_w = max(1, int(round(source.shape[1] * scale)))
    resized_h = max(1, int(round(source.shape[0] * scale)))
    resized = cv2.resize(source, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def _sample_video(
    path: Path,
    target_fps: float,
    min_frames: int,
    max_frames: int,
) -> tuple[np.ndarray, object, dict[str, object]]:
    import cv2
    from transformers.video_utils import VideoMetadata

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or source_fps <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: frames={total}, fps={source_fps}")
    requested = int(total / source_fps * target_fps)
    frame_count = min(min(max(requested, min_frames), max_frames), total)
    indices = np.linspace(0, total - 1, frame_count).round().astype(np.int64)
    selected = set(indices.tolist())
    frames: list[np.ndarray] = []
    index = 0
    while capture.isOpened() and index <= int(indices[-1]):
        ok, bgr = capture.read()
        if not ok:
            break
        if index in selected:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if len(frames) != len(indices):
        raise ValueError(
            f"decoded {len(frames)} selected frames, expected {len(indices)} from {path}"
        )
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=source_fps,
        width=source_w,
        height=source_h,
        duration=total / source_fps,
        video_backend="opencv",
        frames_indices=indices.tolist(),
    )
    report = {
        "source_frames": total,
        "source_fps": source_fps,
        "source_width": source_w,
        "source_height": source_h,
        "sampled_frames": len(frames),
        "sampled_indices": indices.tolist(),
        "target_fps": target_fps,
    }
    return np.stack(frames), metadata, report


def _load_image(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV cannot open image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, {
        "source_width": int(rgb.shape[1]),
        "source_height": int(rgb.shape[0]),
    }


def _numpy(value: object, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--video")
    media.add_argument("--image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--question", default="Describe this video in detail.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.fps <= 0 or args.min_frames <= 0 or args.max_frames < args.min_frames:
        raise ValueError("invalid video sampling limits")
    model_root = Path(args.model_path).resolve()
    contract = ModelContract.from_model(model_root)
    factor = contract.vision_patch_size * contract.spatial_merge_size
    if args.height % factor or args.width % factor:
        raise ValueError(
            f"bucket must be divisible by patch_size*merge_size={factor}"
        )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_root, trust_remote_code=True)
    if args.video:
        kind = "video"
        media_path = Path(args.video).resolve()
        frames, metadata, media_report = _sample_video(
            media_path, args.fps, args.min_frames, args.max_frames
        )
        bucketed = np.stack(
            [_letterbox_rgb(frame, args.height, args.width) for frame in frames]
        )
        content = [
            {"type": "video", "video": str(media_path)},
            {"type": "text", "text": args.question},
        ]
    else:
        kind = "image"
        media_path = Path(args.image).resolve()
        frame, media_report = _load_image(media_path)
        bucketed = _letterbox_rgb(frame, args.height, args.width)
        metadata = None
        content = [
            {"type": "image", "image": str(media_path)},
            {"type": "text", "text": args.question},
        ]
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    common = {
        "text": [prompt],
        # Transformers 4.57's fast Qwen image/video processors only support
        # PT internally. The persisted deployment ABI remains NumPy.
        "return_tensors": "pt",
        "do_resize": False,
    }
    if kind == "video":
        processed = processor(
            videos=[bucketed],
            video_metadata=[metadata],
            do_sample_frames=False,
            **common,
        )
        pixels = _numpy(processed["pixel_values_videos"], np.dtype(np.float32))
        grid = _numpy(processed["video_grid_thw"], np.dtype(np.int64))
        pixel_name = "pixel_values_videos.npy"
        grid_name = "video_grid_thw.npy"
    else:
        processed = processor(images=[bucketed], **common)
        pixels = _numpy(processed["pixel_values"], np.dtype(np.float32))
        grid = _numpy(processed["image_grid_thw"], np.dtype(np.int64))
        pixel_name = "pixel_values.npy"
        grid_name = "image_grid_thw.npy"
    expected_grid = np.asarray(
        [
            (
                (bucketed.shape[0] + contract.temporal_patch_size - 1)
                // contract.temporal_patch_size
                if kind == "video"
                else 1
            ),
            args.height // contract.vision_patch_size,
            args.width // contract.vision_patch_size,
        ]
    )
    if grid.shape != (1, 3) or not np.array_equal(grid[0], expected_grid):
        raise ValueError(f"processor grid {grid.tolist()} != expected {expected_grid.tolist()}")
    expected_patches = int(np.prod(expected_grid))
    if pixels.shape != (expected_patches, contract.patch_vector_size):
        raise ValueError(
            f"pixel tensor {pixels.shape}, expected {(expected_patches, contract.patch_vector_size)}"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / pixel_name, np.ascontiguousarray(pixels))
    np.save(output / grid_name, grid)
    np.save(output / "input_ids.npy", _numpy(processed["input_ids"], np.dtype(np.int64)))
    np.save(
        output / "attention_mask.npy",
        _numpy(processed["attention_mask"], np.dtype(np.int64)),
    )
    manifest = {
        "format": "qwen3-vl-processor-bundle-v1",
        "model_path": str(model_root),
        "kind": kind,
        "media": str(media_path),
        "question": args.question,
        "system": args.system,
        "prompt": prompt,
        "bucket_height": args.height,
        "bucket_width": args.width,
        "grid_thw": grid.tolist(),
        "patches": expected_patches,
        "patch_vector_size": contract.patch_vector_size,
        "visual_tokens": expected_patches // contract.spatial_merge_size**2,
        "valid_seq_len": int(_numpy(processed["attention_mask"], np.dtype(np.int64)).sum()),
        "media_info": media_report,
        "files": {
            "pixels": pixel_name,
            "grid_thw": grid_name,
            "input_ids": "input_ids.npy",
            "attention_mask": "attention_mask.npy",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
