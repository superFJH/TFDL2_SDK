"""CPU reference regression probe; no NPU, checkpoints or model downloads.

Compare builds in separate processes, e.g. TFLLM_CPU_THREADS=8/32 and
TFLLM_TORCH_LIBRARY=/path/to/libtfllm-torch.so. Both dtypes use identical
FP16-representable operands; FP32 calls retain FP32 output arithmetic.
"""
import argparse
import json
import os
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuConv2d, NpuLinear, NpuRuntime


def measure(call, warmup, repeat):
    times = []
    for i in range(warmup + repeat):
        start = time.perf_counter()
        out = call()
        elapsed = (time.perf_counter() - start)*1000
        if i >= warmup:
            times.append(elapsed)
    return out, {"median_ms": statistics.median(times), "samples_ms": times}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--torch-threads", type=int, default=32)
    p.add_argument("--operators", choices=("linear", "conv2d", "both"), default="both")
    p.add_argument("--dtypes", choices=("float16", "float32", "both"), default="both")
    args = p.parse_args()
    if min(args.rows, args.repeat, args.torch_threads) < 1 or args.warmup < 0:
        p.error("rows/repeat/torch-threads must be positive and warmup nonnegative")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(217)
    print("CPU_REFERENCE_CONFIG " + json.dumps({**vars(args), "torch": torch.__version__,
          "environment": {k: os.environ.get(k) for k in ("TFLLM_CPU_THREADS", "TFLLM_TORCH_LIBRARY")}}), flush=True)
    with torch.inference_mode(), NpuRuntime(backend="cpu") as runtime:
        def run(name, shape, x, w, reference, make):
            x32, w32 = x.float(), w.float()
            want, cpu = measure(lambda: reference(x32, w32), args.warmup, args.repeat)
            for dtype in (torch.float16, torch.float32):
                if args.dtypes != "both" and str(dtype).split(".")[-1] != args.dtypes:
                    continue
                value = x.to(dtype)
                layer = make(w.to(dtype), runtime)
                layer.prepare()  # Static weight preparation is outside the timing.
                got, native = measure(lambda: layer(value), args.warmup, args.repeat)
                a, b = got.double().flatten(), want.double().flatten()
                norm = torch.linalg.vector_norm(b).clamp_min(1e-20)
                cos = float(torch.dot(a, b)/(torch.linalg.vector_norm(a).clamp_min(1e-20)*norm))
                print("CPU_REFERENCE_RESULT " + json.dumps({"operator": name, "shape": shape,
                      "dtype": str(dtype), "tfllm_cpu": native, "torch_fp32": cpu,
                      "finite": bool(torch.isfinite(a).all()), "cos": cos,
                      "relative_l2": float(torch.linalg.vector_norm(a-b)/norm)}), flush=True)
                layer.close()
        if args.operators in ("linear", "both"):
            x = torch.randn(args.rows, 768).half()
            w = (torch.randn(2304, 768)*.03).half()
            run("linear", [args.rows, 768, 2304], x, w, F.linear,
                lambda w, r: NpuLinear(w, runtime=r))
        if args.operators in ("conv2d", "both"):
            x = torch.randn(1, 256, 64, 64).half()
            w = (torch.randn(256, 256, 3, 3)*.03).half()
            run("conv2d", [1, 256, 64, 64, 256, 3, 1, 1], x, w,
                lambda x, w: F.conv2d(x, w, padding=1),
                lambda w, r: NpuConv2d(w, padding=1, runtime=r))


if __name__ == "__main__":
    main()
