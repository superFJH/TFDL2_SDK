#!/usr/bin/env python3
"""Compare one or more TFDL Mage-ViT graphs with the PyTorch reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import build_mage_vit as mage


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64, copy=False).ravel()
    got = actual.astype(np.float64, copy=False).ravel()
    error = got - ref
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    denom = ref_norm * got_norm
    return {
        "cosine": float(np.dot(ref, got) / denom) if denom else 0.0,
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "rel_l2": float(np.linalg.norm(error) / ref_norm) if ref_norm else 0.0,
        "reference_norm": ref_norm,
        "actual_norm": got_norm,
    }


def execute_fb(
    fb: Path,
    raw_uint8: np.ndarray,
    sin: np.ndarray,
    cos: np.ndarray,
    addon: Path,
    use_hardware: bool,
) -> tuple[np.ndarray, str, float, float]:
    from TFDL2 import TFContext, TFExecutor
    from TFDL2.utils import LoadCustomOp

    LoadCustomOp(str(addon))
    started = time.perf_counter()
    # The executor shares weights with its context; retain the context for the
    # complete evaluation instead of passing a short-lived temporary object.
    context = TFContext(path=str(fb))
    executor = TFExecutor(
        context,
        mage.vision_executor_config(use_hardware),
    )
    compile_seconds = time.perf_counter() - started
    inputs = executor.GetInputs()
    # Full-Quant graphs expose UINT8 directly; source-Q/DQ QuantizeLite graphs
    # retain the raw 0..255 Placeholder as FP32 and quantize inside the graph.
    pixel_input = (
        raw_uint8.astype(np.float32)
        if "FLOAT" in str(inputs[0].dtype)
        else raw_uint8
    )
    inputs[0].fromNumpy(pixel_input)
    inputs[1].fromNumpy(sin)
    inputs[2].fromNumpy(cos)
    started = time.perf_counter()
    raw_output = executor()[0].toNumpy()
    output_dtype = str(raw_output.dtype)
    output = raw_output.astype(np.float32)
    execute_seconds = time.perf_counter() - started
    return output, output_dtype, compile_seconds, execute_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fb", action="append", required=True)
    parser.add_argument("--canvas-index", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    entry = manifest["canvases"][args.canvas_index]
    canvas_height = int(manifest["canvas_height"])
    canvas_width = int(manifest["canvas_width"])
    config = mage.MageVisionConfig.from_model(
        args.model_path, (canvas_height, canvas_width)
    )
    rgb = mage._read_ppm(bundle / entry["file"])
    raw_uint8 = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    positions = np.asarray(entry["patch_positions"], dtype=np.int64)
    sin, cos = mage.compute_rope(positions, config.head_dim)

    weights = mage.load_vision_weights(args.model_path, config)
    torch_graph = mage.TorchMageVision(config, weights, torch.device("cpu"))
    started = time.perf_counter()
    with torch.no_grad():
        reference = torch_graph(
            torch.from_numpy(raw_uint8), torch.from_numpy(positions)
        ).numpy()
    reference_seconds = time.perf_counter() - started

    report: dict[str, object] = {
        "canvas": entry["file"],
        "canvas_shape": [canvas_height, canvas_width],
        "output_shape": list(reference.shape),
        "pytorch_reference_seconds": reference_seconds,
        "use_hardware": bool(args.hardware),
        "graphs": [],
    }
    for value in args.fb:
        fb = Path(value)
        actual, output_dtype, compile_seconds, execute_seconds = execute_fb(
            fb,
            raw_uint8,
            sin.numpy(),
            cos.numpy(),
            Path(args.addon_path),
            args.hardware,
        )
        item: dict[str, object] = {
            "fb": str(fb),
            "bytes": fb.stat().st_size,
            "dtype": output_dtype,
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
            **metrics(reference, actual),
        }
        report["graphs"].append(item)  # type: ignore[union-attr]
        print(
            f"{fb.name}: cosine={item['cosine']:.6f} "
            f"rel_l2={item['rel_l2']:.6f} execute={execute_seconds:.3f}s "
            f"size={fb.stat().st_size / (1024**2):.1f}MiB"
        )

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
