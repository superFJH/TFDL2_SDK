"""tfllm-vision / python -m tfllm.vision: local segmentation and depth."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from . import load, supported_models


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy() if value.is_floating_point() else value.cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.stack([_array(v) for v in value])
    return np.asarray(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("list", "segment", "depth"))
    parser.add_argument("--family", choices=supported_models())
    parser.add_argument("--checkpoint")
    parser.add_argument("--provider", choices=("auto", "official", "transformers"), default="auto")
    parser.add_argument("--config")
    parser.add_argument("--encoder")
    parser.add_argument("--image")
    parser.add_argument("--output", help="raw masks/scores or depth/cameras in .npz")
    parser.add_argument("--report", help="write backend, elapsed time and operator coverage JSON")
    parser.add_argument("--backend", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--point", type=float, nargs=2, action="append")
    parser.add_argument("--labels", type=int, nargs="+")
    parser.add_argument("--box", type=float, nargs=4)
    parser.add_argument("--text")
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args(argv)
    if args.task == "list":
        print("\n".join(supported_models()))
        return 0
    for key in ("family", "checkpoint", "image", "output"):
        if getattr(args, key) is None:
            parser.error(f"--{key} is required")
    if args.threads <= 0 or args.input_size <= 0:
        parser.error("--threads and --input-size must be positive")
    if args.task == "depth" and not args.family.startswith("depth-"):
        parser.error("depth task requires a Depth Anything family")
    if args.task == "segment" and args.family.startswith("depth-"):
        parser.error("segment task requires a SAM family")
    if args.labels is not None and (not args.point or len(args.labels) != len(args.point)):
        parser.error("--labels must match the number of --point arguments")
    output = Path(args.output)
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    torch.set_num_threads(args.threads)
    with Image.open(args.image) as source:
        image = source.convert("RGB").copy()
    with load(args.family, args.checkpoint, backend=args.backend, chip=args.chip, strict=args.strict,
              provider=args.provider, config=args.config, encoder=args.encoder) as session:
        start = time.perf_counter()
        if args.task == "segment":
            result = session.segment(image if args.family == "sam3" or session.provider == "transformers" else np.asarray(image),
                                     point_coords=None if args.point is None else np.asarray(args.point),
                                     point_labels=None if args.point is None else np.asarray(args.labels or [1]*len(args.point)),
                                     box=None if args.box is None else np.asarray(args.box), text_prompt=args.text)
            arrays = {k: _array(v) for k, v in result.items() if k in {"masks", "scores", "boxes", "low_res_masks"}}
        elif args.family == "depth-anything-v3":
            result = session.inference([str(Path(args.image).resolve())], process_res=args.input_size)
            arrays = {k: _array(getattr(result, k)) for k in ("depth", "conf", "extrinsics", "intrinsics")
                      if getattr(result, k, None) is not None}
        else:
            arrays = {"depth": session.infer_image(np.asarray(image)[..., ::-1].copy(), input_size=args.input_size)}
        elapsed = (time.perf_counter() - start)*1000
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        report = {"family": session.family, "provider": session.provider, "backend": args.backend,
                  "elapsed_ms": elapsed, "coverage": session.report.to_dict()}
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "coverage"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
