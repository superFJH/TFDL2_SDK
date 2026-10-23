"""Board-only NCHW pointwise timing against the previous Linear bridge.

Fixed random operands, no model downloads. Validation is outside timing.
Compares direct FP32 quantization with the previous explicit half/float bridge.
"""
import argparse
import hashlib
import json
import os
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuConv2d, NpuRuntime


def measure(call, warmup, repeat):
    for _ in range(warmup): value = call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        value = call()
        samples.append((time.perf_counter()-start)*1000)
    return value, {'median_ms': statistics.median(samples), 'samples_ms': samples}


def measure_pair(new, old, warmup, repeat):
    calls=(new,old);values=[None,None];samples=[[],[]]
    for trial in range(warmup+repeat):
        for i in (0,1) if trial%2==0 else (1,0):
            start=time.perf_counter();values[i]=calls[i]()
            elapsed=(time.perf_counter()-start)*1000
            if trial>=warmup:samples[i].append(elapsed)
    return values[0],values[1],*({'median_ms':statistics.median(s),'samples_ms':s} for s in samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeat', type=int, default=21)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--small-only', action='store_true')
    a = p.parse_args()
    if min(a.repeat, a.warmup, a.threads) < 1: p.error('counts must be positive')
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    print('POINTWISE_CONFIG '+json.dumps({'threads':a.threads,'repeat':a.repeat,'warmup':a.warmup,
        'verify':os.getenv('TFLLM_NPU_VERIFY_OUTPUTS'),'enabled':os.getenv('TFLLM_NPU_POINTWISE_NCHW'),
        'input_output':'FP16; direct FP32; previous FP32 half/float bridge'}), flush=True)
    shapes = [(64,64,64), (256,256,64), (256,256,128), (128,256,32),
              (112,448,256), (448,112,256)]
    with torch.inference_mode(), NpuRuntime(memory_fraction=.125) as runtime:
        if not hasattr(runtime._lib, 'tfllm_torch_pointwise'):
            raise RuntimeError('rebuild the native pointwise library before benchmarking')
        for ci,co,h in shapes[:1] if a.small_only else shapes:
            torch.manual_seed(217)
            x=torch.randn(1,ci,h,h).half();w=(torch.randn(co,ci,1,1)*.03).half()
            xf,wf=x.float(),w.float()
            layer=NpuConv2d(w,runtime=runtime)
            def legacy(value):
                pixels=value.permute(0,2,3,1).contiguous()
                return runtime.linear(pixels,layer.prepare(),co).permute(0,3,1,2).contiguous()
            got,want,new,old=measure_pair(lambda:layer(x),lambda:legacy(x),a.warmup,a.repeat)
            bridged,old_bridged,new_bridge,old_bridge=measure_pair(
                lambda:layer(xf.half()).float(),lambda:legacy(xf.half()).float(),a.warmup,a.repeat)
            direct,_,direct_stats,bridge_stats=measure_pair(
                lambda:layer(xf),lambda:layer(xf.half()).float(),a.warmup,a.repeat)
            _,cpu=measure(lambda:F.conv2d(xf,wf),a.warmup,a.repeat)
            exact=torch.equal(got,want) and torch.equal(bridged,old_bridged) and torch.equal(direct.half(),got)
            repeat_exact=torch.equal(got,layer(x))
            print('POINTWISE_RESULT '+json.dumps({'shape':{'B':1,'Cin':ci,'Cout':co,'H':h,'W':h},
                'nchw_fp16':new,'legacy_fp16':old,'nchw_fp32_bridge':new_bridge,
                'legacy_fp32_bridge':old_bridge,'cpu_fp32':cpu,'exact':exact,'repeat_exact':repeat_exact,
                'nchw_fp32_direct':direct_stats,'paired_nchw_fp32_bridge':bridge_stats,
                'output_sha256':hashlib.sha256(got.view(torch.uint8).numpy().tobytes()).hexdigest()}),flush=True)
            if not exact or not repeat_exact: raise RuntimeError('pointwise parity failed')
            layer.close()
    print('POINTWISE_DONE',flush=True)


if __name__ == '__main__': main()
