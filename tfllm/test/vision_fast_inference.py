"""Measured NPU40T image inference for the server's official SAM checkpoints.

Example: python3 vision_fast_inference.py sam3 --text cat --verify --output sam3.json
Auto selects the measured per-model backend, including Torch for EfficientTAM-S.
The model stays FP32 outside NPU W8A8/I32 projections. No global FP16 autocast.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


MODELS = {
    "sam2-tiny": ("sam2", "sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "sam2-base": ("sam2", "sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "sam3": ("sam3", "sam3/sam3.pt", None),
    "efficienttam-s512": ("efficienttam", "efficienttam/efficienttam_s_512x512.pt", "configs/efficienttam/efficienttam_s_512x512.yaml"),
}
PROFILES = {
    "sam2-tiny": dict(engine="tfllm", linear_dim=256, linear_rows=512, threads=40),
    "sam2-base": dict(engine="tfllm", linear_dim=256, linear_rows=0, threads=40),
    "sam3": dict(engine="tfllm", linear_dim=128, linear_rows=512, threads=40),
    "efficienttam-s512": dict(engine="torch", linear_dim=0, linear_rows=0, threads=40),
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--sdk-root", type=Path, default=Path("/root/TFDL2_SDK"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--text", default="cat")
    parser.add_argument("--point", type=float, nargs=2, default=(240., 180.))
    parser.add_argument("--engine", choices=("auto", "tfllm", "torch"), default="auto")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--verify", action="store_true", help="also time the original Torch FP32 oracle")
    parser.add_argument("--flip-check", action="store_true", help="verify one additional horizontally flipped input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, help="best mask/score NPZ; default: OUTPUT.mask.npz")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or (args.threads is not None and args.threads < 1):
        parser.error("repeat/threads must be positive and warmup must be nonnegative")
    for vendor in ("pylibs", "sam2-main", "sam3-main", "EfficientTAM-main"):
        sys.path.insert(0, str(args.sdk_root/"vendor"/vendor))
    os.environ.setdefault("TFLLM_CPU_THREADS", "8")
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    from torch import nn
    from tfllm.torch import NpuRuntime
    from tfllm.vision import attach
    from tfllm.vision.loaders import build
    from vision_fast_profile import prepare, guard_short_linears

    profile = dict(PROFILES[args.model])
    if args.threads is not None:
        profile["threads"] = args.threads
    requested_engine = args.engine
    if args.engine == "auto":
        args.engine = "tfllm" if os.environ.get("TFLLM_NPU_ATTENTION_FAST")=="1" else profile["engine"]
    torch.set_num_threads(profile["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(317)
    family, checkpoint, config = MODELS[args.model]
    checkpoint = args.sdk_root/"models"/checkpoint
    started = time.perf_counter()
    model, processor, provider = build(family, checkpoint, **({"config": config} if config else {}))
    load_ms = (time.perf_counter()-started)*1000
    image_path = args.image or args.sdk_root/"bin/data/cat.jpg"
    image = Image.open(image_path).convert("RGB")
    images = {"original": image}
    if args.flip_check:
        images["flipped"] = ImageOps.mirror(image)
    session = None

    def invoke(function, *a, **kw):
        return session._invoke(function, *a, **kw) if session is not None else function(*a, **kw)

    def prompt(state, flipped=False):
        if family == "sam3":
            out = invoke(processor.set_text_prompt, prompt=args.text, state=state)
            masks, scores = torch.as_tensor(out["masks"]), torch.as_tensor(out["scores"])
        else:
            point = list(args.point)
            if flipped:
                point[0] = image.width-1-point[0]
            out = invoke(processor.predict, point_coords=np.array([point], dtype=np.float32),
                         point_labels=np.array([1], dtype=np.int32), multimask_output=True)
            masks, scores = torch.as_tensor(out[0]), torch.as_tensor(out[1])
        if not scores.numel() or not torch.isfinite(scores).all():
            raise RuntimeError("no finite predicted object; choose a prompt present in the image")
        return masks[int(scores.argmax())].squeeze().bool().clone(), scores.float().clone()

    def infer(which="original"):
        pixels = images[which] if family == "sam3" else np.asarray(images[which]).copy()
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            begin = time.perf_counter()
            state = invoke(processor.set_image, pixels)
            encoded = time.perf_counter()
            mask, scores = prompt(state, which == "flipped")
            end = time.perf_counter()
        return dict(encode_ms=(encoded-begin)*1000, prompt_ms=(end-encoded)*1000,
                    total_ms=(end-begin)*1000), mask, scores, state

    def measure():
        for _ in range(args.warmup):
            infer()
        samples = []
        for _ in range(args.repeat):
            timing, mask, scores, state = infer()
            samples.append(timing)
        return dict(median_ms={k: statistics.median(x[k] for x in samples) for k in samples[0]},
                    samples=samples), mask, scores, state

    report = dict(model=args.model, checkpoint=str(checkpoint), profile=profile, engine=args.engine, requested_engine=requested_engine,
                  dtype=("external FP32; NPU W8A8/I32; no expanded U8 GEMM output"
                         if args.engine == "tfllm" else "original Torch FP32"),
                  torch_version=torch.__version__, load_ms=load_ms, repeat=args.repeat, warmup=args.warmup,
                  image=str(image_path), image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  scope="full-resolution set_image + one prompt; no model load/warmup/HTTP in timed samples")
    gold = {}
    if args.verify or args.engine == "torch":
        report["torch"], mask, scores, state = measure()
        result_mask, result_scores = mask, scores
        gold["original"] = mask, scores
        if args.flip_check:
            _, mask, scores, _ = infer("flipped")
            gold["flipped"] = mask, scores
    undo = []
    runtime = None
    try:
        if args.engine == "tfllm":
            undo = prepare(model)
            originals = {id(m): m.forward for m in model.modules() if type(m) is nn.Linear}
            runtime = NpuRuntime(memory_fraction=.125, attention_policy="hybrid",
                                 attention_fuse_workers=32, attention_score_mib=16)
            session = attach(model, family=family, runtime=runtime, processor=processor, provider=provider,
                             predicate=lambda path, m: type(m) is not nn.Linear
                             or min(m.in_features, m.out_features) >= profile["linear_dim"])
            guard_short_linears(model, originals, model._tfllm_optimization, profile["linear_rows"])
            report["tfllm"], mask, scores, state = measure()
            result_mask, result_scores = mask, scores
            cached = []
            with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                for _ in range(args.repeat):
                    begin = time.perf_counter()
                    prompt(state)
                    cached.append((time.perf_counter()-begin)*1000)
            report["cached_prompt_ms"] = dict(median=statistics.median(cached), samples=cached)
            if gold:
                quality = {}
                for which, (reference, ref_scores) in gold.items():
                    if which != "original":
                        _, mask, scores, _ = infer(which)
                    union = int((mask | reference).sum())
                    quality[which] = dict(mask_iou=int((mask & reference).sum())/union if union else 1.,
                        max_score_delta=float((scores-ref_scores).abs().max()) if scores.shape == ref_scores.shape else None,
                        predictions=scores.numel(), reference_predictions=ref_scores.numel())
                report["quality"] = quality
            report["coverage"] = session.report.to_dict()
            report["memory"] = runtime.memory_stats()
            library = Path(runtime._lib._name)
            report["library"] = str(library)
            if library.is_file():
                report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        if args.engine == "torch":
            if args.flip_check:
                _, _, _, state = infer()
            cached = []
            with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                for _ in range(args.repeat):
                    begin = time.perf_counter()
                    prompt(state)
                    cached.append((time.perf_counter()-begin)*1000)
            report["cached_prompt_ms"] = dict(median=statistics.median(cached), samples=cached)
        report["inference"] = report[args.engine]
    finally:
        if session is not None:
            session.close()
        for restore in reversed(undo):
            restore()
        if runtime is not None:
            runtime.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_path = args.mask_output or args.output.with_suffix(".mask.npz")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, mask=result_mask.numpy(), score=result_scores.max().numpy())
    report["mask_output"] = str(mask_path)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k in
                     ("model", "engine", "profile", "torch", "tfllm", "quality", "cached_prompt_ms")}, indent=2))


if __name__ == "__main__":
    main()
