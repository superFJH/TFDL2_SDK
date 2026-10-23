"""Board operator timing with FP32 CPU baselines and native FP32 bridge timing.

Fixed random operands; no model downloads. Run each tile/cache configuration in
its own process. Timings include synchronous completion and exclude validation.
"""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuConv2d, NpuLinear, NpuRuntime
from tfllm.torch.runtime import _library


def measure(call, warmup, repeat, check_repeat=False):
    start = time.perf_counter()
    value = call()
    cold = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        value = call()
    previous = value
    stable = True
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        value = call()
        times.append((time.perf_counter() - start) * 1000)
        if check_repeat:
            stable = stable and torch.equal(value, previous)
        previous = value
    return value, {'cold_ms': cold, 'median_ms': statistics.median(times),
                   'min_ms': min(times), 'max_ms': max(times), 'samples_ms': times,
                   'repeat_bit_equal': stable if check_repeat else None}


def metric(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if not bool(torch.isfinite(a).all()):
        return {'finite': False}
    na = torch.linalg.vector_norm(a, dtype=torch.float64).clamp_min(1e-20)
    nb = torch.linalg.vector_norm(b, dtype=torch.float64).clamp_min(1e-20)
    return {'finite': True, 'cos': float(((a*b).sum(dtype=torch.float64)/(na*nb)).clamp(-1, 1)),
            'relative_l2': float(torch.linalg.vector_norm(a-b, dtype=torch.float64)/nb),
            'max_abs_error': float((a-b).abs().max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--select', default='all')
    parser.add_argument('--no-bridge', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=0.125)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    lib = pathlib.Path(_library(None)._name).resolve()
    print('COLLECT_CONFIG ' + json.dumps({'torch_version': torch.__version__,
        'torch_threads': args.threads, 'interop_threads': 1, 'seed': 217,
        'warmup': args.warmup, 'repeat': args.repeat, 'memory_fraction': args.memory_fraction, 'library': str(lib),
        'library_sha256': hashlib.sha256(lib.read_bytes()).hexdigest(),
        'cpu_baseline': 'FP16-rounded operands promoted to FP32 outside timing',
        'npu_entrance': 'automatic FP16/FP32 external tensors; native W8A8 arithmetic',
        'env': {n: os.environ.get(n) for n in ['TFLLM_NPU_ADAPTIVE_SPLIT', 'TFLLM_NPU_MAX_CORES',
            'TFLLM_NPU_SPLIT_MIN_MACS', 'TFLLM_NPU_TRACE_SPLIT', 'TFLLM_NPU_VERIFY_OUTPUTS',
            'TFLLM_CPU_THREADS', 'TFLLM_NPU_ATTENTION_FUSE_WORKERS',
            'TFLLM_NPU_LINEAR_TILE_ROWS', 'TFLLM_NPU_LINEAR_PIPELINE',
            'TFLLM_NPU_WINDOW_BATCH', 'TFLLM_NPU_CONV_WORKSPACE_REUSE']} }), flush=True)
    selected = set(args.select.split(','))

    def wanted(name):
        return 'all' in selected or name in selected

    def record(name, shape, native, cpu, bridge, flops):
        a, npu = measure(native, args.warmup, args.repeat, True)
        b, host = measure(cpu, args.warmup, args.repeat)
        bridged = None
        if not args.no_bridge:
            bridge_value, bridged = measure(bridge, args.warmup, args.repeat, True)
        print('COLLECT_RESULT ' + json.dumps({'name': name, 'shape': shape,
              'npu_fp16': npu, 'torch_fp32': host, 'fp32_bridge_to_npu': bridged,
              'cpu_over_npu': host['median_ms']/npu['median_ms'],
              'logical_gflops': flops/1e9,
              'npu_logical_gflops_per_second': flops/npu['median_ms']/1e6,
              'vs_fp32': metric(a, b),
              'output_sha256': hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
              'fp32_vs_fp16': metric(bridge_value, a.float()) if bridged is not None else None,
              'fp32_vs_torch': metric(bridge_value, b) if bridged is not None else None}), flush=True)

    with torch.inference_mode(), NpuRuntime(memory_fraction=args.memory_fraction) as runtime:
        linears = [('qkv_large',4096,768,2304), ('fc1_large',4096,768,3072),
                   ('fc2_large',4096,3072,768), ('gemm_2048',2048,2048,2048),
                   ('fc1_hiera',65536,112,448), ('fc2_hiera',65536,448,112),
                   ('qkv_hiera',65536,112,336)]
        for name, m, k, n in linears:
            if not wanted(name): continue
            torch.manual_seed(217)
            x, w = torch.randn(m,k).half(), (torch.randn(n,k)*.03).half()
            xf, wf = x.float(), w.float()
            layer = NpuLinear(w, runtime=runtime)
            record(name, {'M':m,'K':k,'N':n}, lambda:layer(x), lambda:F.linear(xf,wf),
                   lambda:layer.forward_fp32(xf), 2*m*k*n)
            layer.close()
            del layer, x, w, xf, wf
        convs = [('patch_embed',1024,3,96,7,4),
                 ('conv_3x3_rgb64',64,3,96,3,1), ('conv_3x3_rgb128',128,3,96,3,1),
                 ('neck_1x1_256',64,256,256,1,1)]
        for name, h, ci, co, kernel, stride in convs:
            if not wanted(name): continue
            torch.manual_seed(217)
            x, w = torch.randn(1,ci,h,h).half(), (torch.randn(co,ci,kernel,kernel)*.03).half()
            xf, wf = x.float(), w.float()
            layer = NpuConv2d(w,stride=stride,padding=kernel//2,runtime=runtime)
            oh = (h+2*(kernel//2)-kernel)//stride+1
            record(name, {'B':1,'H':h,'W':h,'Cin':ci,'Cout':co,'kernel':kernel,'stride':stride},
                   lambda:layer(x), lambda:F.conv2d(xf,wf,stride=stride,padding=kernel//2),
                   lambda:layer(xf), 2*oh*oh*ci*co*kernel*kernel)
            layer.close()
            del layer, x, w, xf, wf
        attention = [('window_1024_d64',1024,1,64,64,64),
                     ('window_1024_h4_d64',1024,4,64,64,64),
                     ('window_256_d192',256,1,64,64,192),
                     ('window_256_h4_d192',256,4,64,64,192),
                     ('window_256_h3_d64',256,3,64,64,64),
                     ('memory_cross',1,1,4096,64,64),
                     ('attention_2048',1,1,2048,2048,128),
                     ('attention_2048_h4',1,4,2048,2048,128)]
        for name, batch, heads, queries, keys, dim in attention:
            if not wanted(name): continue
            torch.manual_seed(217)
            q=torch.randn(batch,heads,queries,dim).half()
            k=torch.randn(batch,heads,keys,dim).half();v=torch.randn_like(k)
            qf,kf,vf=q.float(),k.float(),v.float()
            record(name, {'B':batch,'heads':heads,'Q':queries,'KV':keys,'dim':dim},
                   lambda:runtime.attention(q,k,v), lambda:F.scaled_dot_product_attention(qf,kf,vf),
                   lambda:runtime.attention_fp32(qf,kf,vf),
                   4*batch*heads*queries*keys*dim)
            del q,k,v,qf,kf,vf
    print('COLLECT_DONE', flush=True)


if __name__ == '__main__':
    main()
