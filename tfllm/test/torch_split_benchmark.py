"""Board-only warm operator timing, fixed random operands, no model downloads.

Run in separate processes with adaptive splitting on/off or different core caps.
Normal TFLLM CPU mode and fake SDK timings are not hardware performance results.
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
    start = time.perf_counter()
    result = call()
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        result = call()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = call()
        times.append((time.perf_counter() - start) * 1000)
    return result, {"cold_ms": cold_ms, "median_ms": statistics.median(times),
                    "min_ms": min(times), "max_ms": max(times)}


def compare(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    if not bool(torch.isfinite(a).all()):
        return {"finite": False}
    norm = torch.linalg.vector_norm(b, dtype=torch.float64).clamp_min(1e-20)
    anorm = torch.linalg.vector_norm(a, dtype=torch.float64).clamp_min(1e-20)
    return {"finite": True, "cos": float(((a * b).sum(dtype=torch.float64) / (norm * anorm)).clamp(-1, 1)),
            "relative_l2": float(torch.linalg.vector_norm(a - b, dtype=torch.float64) / norm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.torch_threads < 1:
        parser.error("repeat/torch-threads must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(217)
    print("SPLIT_CONFIG " + json.dumps({"torch_threads": args.torch_threads,
        "environment": {name: os.environ.get(name) for name in
            ("TFLLM_NPU_ADAPTIVE_SPLIT", "TFLLM_NPU_MAX_CORES", "TFLLM_NPU_SPLIT_MIN_MACS",
             "TFLLM_NPU_VERIFY_OUTPUTS", "TFLLM_CPU_THREADS")}}))

    def run(name, shape, native, reference):
        got, npu = measure(native, args.warmup, args.repeat)
        want, cpu = measure(reference, args.warmup, args.repeat)
        print("SPLIT_RESULT " + json.dumps({"operator": name, "shape": shape, "npu": npu,
              "torch_cpu": cpu, "vs_torch_fp16": compare(got, want)}), flush=True)

    with torch.inference_mode(), NpuRuntime() as runtime:
        linear_shapes = [(16, 64, 512), (64, 256, 1024), (256, 1024, 1024), (1024, 1024, 4096)]
        for m, k, n in linear_shapes[:1] if args.small_only else linear_shapes:
            x = torch.randn(m, k).half()
            w = (torch.randn(n, k) * .03).half()
            layer = NpuLinear(w, runtime=runtime)
            run("linear", {"M": m, "K": k, "N": n}, lambda: layer(x), lambda: F.linear(x, w))
            del layer

        conv_shapes = [(16, 3, 32, 3, 1), (64, 3, 96, 3, 1),
                       (256, 3, 96, 7, 4), (1024, 3, 96, 7, 4)]
        for h, ci, co, kernel, stride in conv_shapes[:1] if args.small_only else conv_shapes:
            x = torch.randn(1, ci, h, h).half()
            w = (torch.randn(co, ci, kernel, kernel) * .03).half()
            layer = NpuConv2d(w, stride=stride, padding=kernel // 2, runtime=runtime)
            run("conv2d", {"H": h, "W": h, "Cin": ci, "Cout": co, "kernel": kernel, "stride": stride},
                lambda: layer(x), lambda: F.conv2d(x, w, stride=stride, padding=kernel // 2))
            del layer

        attention_shapes = [(16, 64, 64), (64, 256, 64), (256, 1024, 64), (1024, 1024, 128)]
        for qrows, keys, dim in attention_shapes[:1] if args.small_only else attention_shapes:
            q = torch.randn(1, 1, qrows, dim).half()
            k = torch.randn(1, 1, keys, dim).half()
            v = torch.randn_like(k)
            run("sdpa", {"batch": 1, "heads": 1, "Q": qrows, "KV": keys, "dim": dim},
                lambda: runtime.attention(q, k, v), lambda: F.scaled_dot_product_attention(q, k, v))


if __name__ == "__main__":
    main()
