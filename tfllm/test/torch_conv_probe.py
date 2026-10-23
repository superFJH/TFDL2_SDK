"""Offline board Conv reproducer: fixed operands, affine U8 oracle, repeated calls.

Uses random FP16 weights and RGB-like inputs; no checkpoint download. Fake SDK
can validate the test itself but cannot reproduce hardware/cache failures.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuConv2d, NpuRuntime


def round_away(x):
    return x.sign() * (x.abs() + .5).floor()


def affine(x, dims):
    x = x.float()
    lo = x.amin(dims, keepdim=True).clamp(max=0)
    hi = x.amax(dims, keepdim=True).clamp(min=0)
    scale = torch.where(hi == lo, 1., (hi - lo) / 255.)
    zero = round_away(-lo / scale).clamp(0, 255)
    codes = (round_away(x / scale) + zero).clamp(0, 255)
    return codes - zero, scale, zero


def symmetric(x):
    x = x.float()
    peak = x.abs().amax(-1, keepdim=True)
    scale = torch.where(peak == 0, 1., peak / 127.)
    # Linear's existing symmetric kernel multiplies by the reciprocal scale.
    return round_away(x * scale.reciprocal()).clamp(-127, 127), scale


def oracle(x, weight, bias, kernel, stride, pad, spatial_impl="native"):
    if kernel == 1 or spatial_impl == "gemm":
        if kernel == 1:
            pixels = x[:, :, ::stride, ::stride].permute(0, 2, 3, 1).contiguous()
        else:
            pixels = F.unfold(x.float(), kernel, padding=pad, stride=stride).transpose(1, 2).contiguous()
        a, sa = symmetric(pixels)
        w, sw = symmetric(weight.flatten(1))
        result = (F.linear(a, w) * (sa * sw.t())).half()
        if kernel == 1:
            y = result.permute(0, 3, 1, 2).contiguous()
        else:
            oh = (x.shape[2] + 2*pad-kernel)//stride+1
            ow = (x.shape[3] + 2*pad-kernel)//stride+1
            y = result.transpose(1, 2).reshape(x.shape[0], weight.shape[0], oh, ow)
        quantization = {"scheme": "symmetric_per_patch_input_per_output_weight"}
    else:
        a, sa, za = affine(x, (0, 1, 2, 3))
        w, sw, zw = affine(weight, (1, 2, 3))
        # K<=147 here: even the sum of absolute U8 products is <2**24,
        # so FP32 convolution can represent these integer sums exactly.
        y = (F.conv2d(a, w, stride=stride, padding=pad) *
             (sa * sw.reshape(1, -1, 1, 1))).half()
        quantization = {"scheme": "affine_global_input_per_output_weight",
                        "input_scale": sa.item(), "input_zero": int(za.item()),
                        "weight_zero_min": int(zw.min()), "weight_zero_max": int(zw.max())}
    return y + bias[None, :, None, None], quantization


def metrics(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    if not bool(torch.isfinite(a).all()):
        return {"finite": False, "nonfinite": int((~torch.isfinite(a)).sum())}
    delta = a - b
    # Long FP32 dot/norm reductions can report cosine >1 even for equal
    # tensors. Accumulate metrics in FP64 without storing FP64 activations.
    norm = torch.linalg.vector_norm(b, dtype=torch.float64).clamp_min(1e-20)
    a_norm = torch.linalg.vector_norm(a, dtype=torch.float64).clamp_min(1e-20)
    cosine = ((a * b).sum(dtype=torch.float64) / (a_norm * norm)).clamp(-1, 1)
    return {"finite": True, "cos": float(cosine),
            "relative_l2": float(torch.linalg.vector_norm(delta, dtype=torch.float64) / norm),
            "max_abs_error": float(delta.abs().max()),
            "max_abs_output": float(a.abs().max()),
            "peak_ratio": float(a.abs().max() / b.abs().max().clamp_min(1e-20))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--out-channels", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8, help="Torch reference/retained operator threads")
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--library", help="production libtfllm-torch path; defaults to TFLLM_TORCH_LIBRARY")
    parser.add_argument("--controls", action="store_true", help="also run 1x1 and 3x3, all stride=4")
    parser.add_argument("--replay", help="load exact operands from a previous --dump .pt file")
    parser.add_argument("--stride", type=int, help="override stride (1..7), including when replaying")
    parser.add_argument("--spatial-impl", choices=("native", "gemm"), default="native")
    parser.add_argument("--dump", help="optional .pt file for the first failure's input/weights/outputs")
    parser.add_argument("--seed", type=int, default=217)
    args = parser.parse_args(argv)
    if min(args.height, args.width, args.out_channels, args.repeat, args.threads) <= 0:
        parser.error("dimensions, repeat and threads must be positive")
    if args.stride is not None and not 1 <= args.stride <= 7:
        parser.error("--stride must be 1..7")
    if args.replay and args.controls:
        parser.error("--replay preserves one kernel; omit --controls")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    replay = None if args.replay is None else torch.load(args.replay, map_location="cpu", weights_only=True)
    x = replay["input"] if replay is not None else (torch.rand(1, 3, args.height, args.width) * 4.71 - 2.07).half()
    kernels = [int(replay["kernel"])] if replay is not None else ([1, 3, 7] if args.controls else [7])
    if replay is not None:
        weight, bias = replay["weight"], replay["bias"]
        if (x.dtype != torch.float16 or x.ndim != 4 or x.shape[1] != 3 or
                weight.dtype != torch.float16 or weight.ndim != 4 or weight.shape[1] != 3 or
                kernels[0] not in (1, 3, 7) or weight.shape[2:] != (kernels[0], kernels[0]) or
                bias.dtype != torch.float16 or bias.shape != (weight.shape[0],)):
            parser.error("--replay requires this probe's FP16 RGB input, square 1/3/7 kernel and bias")
    config = {"input_shape": list(x.shape), "out_channels": weight.shape[0] if replay is not None else args.out_channels,
              "input_min": float(x.min()), "input_max": float(x.max()),
              "torch_threads": torch.get_num_threads(), "seed": args.seed,
              "replay": args.replay, "spatial_impl": args.spatial_impl}
    print("CONV_CONFIG " + json.dumps(config), flush=True)
    failed, dumped = False, False
    with NpuRuntime(chip=args.chip, backend="npu", library=args.library,
                    spatial_convolution=args.spatial_impl) as runtime, torch.inference_mode():
        for kernel in kernels:
            stride = args.stride if args.stride is not None else (int(replay["stride"]) if replay is not None else 4)
            pad = int(replay["pad"]) if replay is not None else kernel // 2
            if replay is None:
                weight = (torch.randn(args.out_channels, 3, kernel, kernel) * .03).half()
                bias = (torch.randn(args.out_channels) * .01).half()
            expected, quantization = oracle(x, weight, bias, kernel, stride, pad, args.spatial_impl)
            floating = F.conv2d(x.float(), weight.float(), bias.float(), stride=stride, padding=pad)
            print("CONV_ORACLE " + json.dumps({"kernel": kernel, "stride": stride, "pad": pad,
                  "shape": list(expected.shape), "quantization": quantization,
                  "quantized_vs_float": metrics(expected, floating)}), flush=True)
            layer = NpuConv2d(weight, bias, stride=stride, padding=pad, runtime=runtime)
            first = None
            try:
                for trial in range(args.repeat):
                    start = time.perf_counter()
                    actual = layer(x)
                    elapsed = (time.perf_counter() - start) * 1000
                    comparison = metrics(actual, expected)
                    repeat_delta = 0. if first is None else float((actual.float() - first.float()).abs().max())
                    stable = first is None or torch.equal(actual, first)
                    # Allow a few FP16 ulps from different CPU/NEON scale ordering.
                    matches = torch.allclose(actual.float(), expected.float(), rtol=.002, atol=.002)
                    ok = comparison["finite"] and stable and matches
                    print("CONV_RUN " + json.dumps({"kernel": kernel, "stride": stride, "trial": trial,
                          "cold": trial == 0, "ms": elapsed, "vs_quantized": comparison,
                          "repeat_bit_equal": stable, "repeat_max_abs": repeat_delta, "ok": ok}), flush=True)
                    failed |= not ok
                    if first is None:
                        first = actual.clone()
                    if not ok and args.dump and not dumped:
                        path = Path(args.dump);path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"config": config, "kernel": kernel, "stride": stride, "pad": pad,
                                    "input": x, "weight": weight, "bias": bias,
                                    "expected": expected, "actual": actual}, path)
                        dumped = True
                        print("CONV_DUMP " + str(path), flush=True)
            finally:
                layer.close()
            del layer, expected, floating, first
    print("CONV_SUMMARY " + json.dumps({"ok": not failed}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
