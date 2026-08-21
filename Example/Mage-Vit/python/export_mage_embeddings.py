#!/usr/bin/env python3
"""Export or merge Mage-ViT embedding bundles for Qwen A/B evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import build_mage_vit as mage


def canvas_inputs(
    source: Path,
    entry: dict,
    config: mage.MageVisionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = mage._read_ppm(source / entry["file"])
    raw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    positions = np.asarray(entry["patch_positions"], dtype=np.int64)
    sin, cos = mage.compute_rope(positions, config.head_dim)
    return raw, positions, sin.numpy(), cos.numpy()


def export_torch(
    args: argparse.Namespace,
    source: Path,
    manifest: dict,
    indices: list[int],
    config: mage.MageVisionConfig,
) -> list[np.ndarray]:
    device = torch.device(args.device)
    weights = mage.load_vision_weights(args.model_path, config)
    graph = mage.TorchMageVision(config, weights, device)
    outputs: list[np.ndarray] = []
    for completed, index in enumerate(indices, 1):
        raw, positions, _, _ = canvas_inputs(
            source, manifest["canvases"][index], config
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = graph(
                torch.from_numpy(raw).to(device),
                torch.from_numpy(positions).to(device),
            )
        outputs.append(output.float().cpu().numpy())
        print(
            f"[torch] canvas={index} {completed}/{len(indices)} "
            f"seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return outputs


def export_tfdl(
    args: argparse.Namespace,
    source: Path,
    manifest: dict,
    indices: list[int],
    config: mage.MageVisionConfig,
) -> list[np.ndarray]:
    from TFDL2 import TFContext, TFExecutor
    from TFDL2.utils import LoadCustomOp

    if not args.fb:
        raise ValueError("TFDL export requires --fb")
    LoadCustomOp(args.addon_path)
    # CompileExecutor uses shareWeight=true, so the context must remain alive
    # for every canvas executed by this executor.
    context = TFContext(path=args.fb)
    executor = TFExecutor(
        context,
        mage.vision_executor_config(bool(args.hardware)),
    )
    outputs: list[np.ndarray] = []
    for completed, index in enumerate(indices, 1):
        raw, _, sin, cos = canvas_inputs(
            source, manifest["canvases"][index], config
        )
        inputs = executor.GetInputs()
        inputs[0].fromNumpy(
            raw.astype(np.float32)
            if "FLOAT" in str(inputs[0].dtype)
            else raw
        )
        inputs[1].fromNumpy(sin)
        inputs[2].fromNumpy(cos)
        started = time.perf_counter()
        output = executor()[0].toNumpy()
        outputs.append(output.astype(np.float32))
        print(
            f"[tfdl] canvas={index} {completed}/{len(indices)} "
            f"dtype={output.dtype} seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return outputs


def write_bundle(
    output: Path,
    source_manifest: dict,
    indices: list[int],
    embeddings: list[np.ndarray],
    embedding_source: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in indices:
        entry = dict(source_manifest["canvases"][index])
        entry["source_canvas_index"] = index
        entries.append(entry)
    manifest = dict(source_manifest)
    manifest["canvases"] = entries
    manifest["embedding_source"] = embedding_source
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    values = np.concatenate(embeddings, axis=1).reshape(-1)
    values.astype(np.float32).tofile(output / "visual_embeddings.f32")


def export_command(args: argparse.Namespace) -> None:
    source = Path(args.bundle)
    manifest = json.loads((source / "manifest.json").read_text())
    count = len(manifest["canvases"])
    indices = args.indices if args.indices is not None else list(range(count))
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--indices must contain unique canvas ids")
    if min(indices) < 0 or max(indices) >= count:
        raise IndexError(f"canvas index must be in [0,{count})")
    config = mage.MageVisionConfig.from_model(
        args.model_path,
        (int(manifest["canvas_height"]), int(manifest["canvas_width"])),
    )
    if args.backend == "torch":
        embeddings = export_torch(args, source, manifest, indices, config)
    else:
        embeddings = export_tfdl(args, source, manifest, indices, config)
    write_bundle(
        Path(args.output_bundle), manifest, indices, embeddings,
        f"{args.backend}:{args.fb or args.model_path}",
    )


def merge_command(args: argparse.Namespace) -> None:
    source = Path(args.bundle)
    source_manifest = json.loads((source / "manifest.json").read_text())
    by_index: dict[int, tuple[dict, np.ndarray]] = {}
    for shard_value in args.shard:
        shard = Path(shard_value)
        manifest = json.loads((shard / "manifest.json").read_text())
        raw = np.fromfile(shard / "visual_embeddings.f32", dtype=np.float32)
        offset = 0
        for entry in manifest["canvases"]:
            index = int(entry["source_canvas_index"])
            tokens = len(entry["patch_positions"]) // 4
            values = tokens * args.hidden_size
            if index in by_index:
                raise ValueError(f"duplicate canvas {index} across shards")
            by_index[index] = (
                entry,
                raw[offset : offset + values].reshape(1, tokens, args.hidden_size),
            )
            offset += values
        if offset != raw.size:
            raise ValueError(f"unused values in shard {shard}")
    expected = list(range(len(source_manifest["canvases"])))
    if sorted(by_index) != expected:
        raise ValueError(
            f"shards cover {sorted(by_index)}, expected canvas ids {expected}"
        )
    write_bundle(
        Path(args.output_bundle),
        source_manifest,
        expected,
        [by_index[index][1] for index in expected],
        "merged:" + ",".join(args.shard),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--model-path", required=True)
    export.add_argument("--bundle", required=True)
    export.add_argument("--backend", choices=("torch", "tfdl"), required=True)
    export.add_argument("--fb", default=None)
    export.add_argument("--device", default="cpu")
    export.add_argument("--indices", type=int, nargs="+", default=None)
    export.add_argument("--hardware", action="store_true")
    export.add_argument("--output-bundle", required=True)
    export.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    export.set_defaults(handler=export_command)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--bundle", required=True)
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--hidden-size", type=int, default=2560)
    merge.add_argument("--output-bundle", required=True)
    merge.set_defaults(handler=merge_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
