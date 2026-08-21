#!/usr/bin/env python3
"""Convert codec-video-prep canvases to the Mage-Vit frontend bundle ABI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def block_order(positions: np.ndarray, height: int, width: int, merge: int) -> np.ndarray:
    grid_h = height // 16
    grid_w = width // 16
    expected = (grid_h * grid_w, 3)
    if positions.shape != expected:
        raise ValueError(f"patch positions {positions.shape}, expected {expected}")
    return (
        positions.reshape(grid_h // merge, merge, grid_w // merge, merge, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 3)
    )


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"invalid RGB canvas {rgb.shape} {rgb.dtype}")
    with path.open("wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(rgb.tobytes())


def main() -> None:
    args = parse_args()
    source = Path(args.codec_dir)
    output = Path(args.output_dir)
    meta = json.loads((source / "meta.json").read_text())
    positions = np.load(source / "src_patch_position.npy")
    height, width = (int(value) for value in meta["processed_hw"])
    patch_size = int(meta["patch"])
    merge = int(meta["block_size"])
    if patch_size != 16 or merge != 2:
        raise ValueError(
            f"Mage-Vit requires patch=16/block_size=2, got {patch_size}/{merge}"
        )
    files = [source / value for value in meta["canvas_files"]]
    patches_per_canvas = (height // patch_size) * (width // patch_size)
    if positions.shape != (len(files) * patches_per_canvas, 3):
        raise ValueError(
            f"all patch positions {positions.shape}, expected "
            f"{(len(files) * patches_per_canvas, 3)}"
        )

    output.mkdir(parents=True, exist_ok=True)
    fps = float(meta["fps"])
    canvases: list[dict[str, object]] = []
    for index, source_file in enumerate(files):
        rgb = np.asarray(Image.open(source_file).convert("RGB"), dtype=np.uint8)
        if rgb.shape != (height, width, 3):
            raise ValueError(f"{source_file}: RGB shape {rgb.shape}")
        filename = f"canvas_{index:03d}.ppm"
        write_ppm(output / filename, rgb)
        start = index * patches_per_canvas
        ordered = block_order(
            positions[start : start + patches_per_canvas], height, width, merge
        )
        timestamps = ordered[:, 0].astype(np.float64) / fps
        canvases.append(
            {
                "file": filename,
                "timestamp_seconds": float(np.median(timestamps)),
                "token_count": patches_per_canvas // (merge * merge),
                "patch_positions": ordered.tolist(),
                "patch_timestamps": timestamps.tolist(),
            }
        )

    manifest = {
        "format": "megavit.frontend.v1",
        "source": "codec-video-prep-0.2.5",
        "source_video": meta.get("video"),
        "source_video_hash": meta.get("video_hash"),
        "patch_size": patch_size,
        "spatial_merge_size": merge,
        "canvas_width": width,
        "canvas_height": height,
        "canvases": canvases,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "codec_dir": str(source),
                "output_dir": str(output),
                "canvases": len(canvases),
                "visual_tokens": sum(int(x["token_count"]) for x in canvases),
                "canvas_shape": [height, width],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
