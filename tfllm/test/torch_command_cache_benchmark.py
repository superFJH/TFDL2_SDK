"""Warm forward timing, no downloads. Run each cache setting in a fresh process.

TFLLM_NPU_FORWARD_CACHE=0 python3 torch_command_cache_benchmark.py
TFLLM_NPU_FORWARD_CACHE=1 python3 torch_command_cache_benchmark.py
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuConv2d, NpuLinear, NpuRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--dtype', choices=('fp16', 'fp32'), default='fp32')
    parser.add_argument('--select', default='')
    parser.add_argument('--profile-dir', default='', help='Requires a library with the optional probe_profile bridge')
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.threads < 1:
        parser.error('repeat/threads must be positive; warmup must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(217)
    dtype = torch.float32 if args.dtype == 'fp32' else torch.float16
    print('CACHE_CONFIG ' + json.dumps({'torch': torch.__version__, **vars(args), 'env': {
        k: os.getenv(k) for k in ('TFLLM_NPU_FORWARD_CACHE', 'TFLLM_NPU_CACHE_IDLE_MS',
                                 'TFLLM_NPU_WORKSPACE_CACHE_MIB', 'TFLLM_NPU_TEMPLATE_CACHE_MIB')}}), flush=True)
    bridge = None
    if args.profile_dir:
        Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
        bridge = ctypes.CDLL(os.environ['TFLLM_TORCH_LIBRARY'])
        bridge.probe_profile_end.argtypes = [ctypes.c_char_p]

    def measure(call):
        start = time.perf_counter(); result = call()
        cold = (time.perf_counter() - start) * 1000
        for _ in range(args.warmup): result = call()
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter(); result = call()
            samples.append((time.perf_counter() - start) * 1000)
        return result, {'cold_ms': cold, 'median_ms': statistics.median(samples), 'samples_ms': samples}

    def run(name, call, reference):
        result, timing = measure(call)
        assert torch.equal(result, call()), name + ': repeated output changed'
        ref = reference().float().flatten().double()
        out = result.float().flatten().double()
        assert bool(out.isfinite().all()), name + ': non-finite output'
        if bridge:
            bridge.probe_profile_begin(); call()
            bridge.probe_profile_end(str(Path(args.profile_dir) / (name + '.csv')).encode())
        print('CACHE_RESULT ' + json.dumps({'name': name, **timing,
            'output_sha256': hashlib.sha256(result.contiguous().numpy().tobytes()).hexdigest(),
            'cos': float(out.dot(ref) / (out.norm() * ref.norm()).clamp_min(1e-20)),
            'relative_l2': float((out-ref).norm() / ref.norm().clamp_min(1e-20)), 'stable': True}), flush=True)

    def selected(name): return not args.select or args.select in name

    with torch.inference_mode(), NpuRuntime(memory_fraction=.125, attention_fuse_workers=32,
                                            attention_score_mib=16) as runtime:
        for name, m, k, n in [('linear_qkv',4096,768,2304), ('linear_hiera_fc1',65536,112,448),
                              ('linear_hiera_fc2',65536,448,112)]:
            if not selected(name): continue
            x = torch.randn(m,k,dtype=dtype); w = torch.randn(n,k,dtype=dtype)*.03
            layer = NpuLinear(w, runtime=runtime)
            run(name, lambda: layer(x), lambda: F.linear(x,w))
            layer.close()
        if selected('conv_neck'):
            x = torch.randn(1,256,64,64,dtype=dtype); w = torch.randn(256,256,1,1,dtype=dtype)*.03
            layer = NpuConv2d(w,runtime=runtime)
            run('conv_neck',lambda:layer(x),lambda:F.conv2d(x,w))
            layer.close()
        for name,b,h,q,k,d in [('attention_window',1024,1,64,64,64), ('attention_global',1,4,4096,4096,96)]:
            if not selected(name): continue
            query = torch.randn(b,h,q,d,dtype=dtype); key = torch.randn(b,h,k,d,dtype=dtype)
            value = torch.randn_like(key)
            run(name,lambda:runtime.attention(query,key,value),lambda:F.scaled_dot_product_attention(query,key,value))


if __name__ == '__main__':
    main()
