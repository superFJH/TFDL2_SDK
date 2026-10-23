"""A/B public activation calls, including allocation and Python/C ABI overhead."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuRuntime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--backend', choices=('cpu', 'npu'), default='cpu')
    parser.add_argument('--operators', nargs='+', choices=['gelu_erf', 'gelu_tanh', 'silu', 'sigmoid', 'tanh', 'mish', 'softplus', 'elu', 'selu'])
    parser.add_argument('--sizes', nargs='+', type=int, default=[4096, 65536, 1048576, 12582912])
    parser.add_argument('--repeat', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # These control independent pools; set before loading/initializing native.
    os.environ['TFLLM_CPU_THREADS'] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(291)
    rows = []
    with NpuRuntime(backend=args.backend) as runtime, torch.inference_mode():
        ops = [('gelu_erf', runtime.gelu, F.gelu),
               ('gelu_tanh', lambda x: runtime.gelu(x, approximate='tanh'), lambda x: F.gelu(x, approximate='tanh')),
               ('silu', runtime.silu, F.silu), ('sigmoid', runtime.sigmoid, torch.sigmoid),
               ('tanh', runtime.tanh, torch.tanh), ('mish', runtime.mish, F.mish),
               ('softplus', runtime.softplus, F.softplus), ('elu', runtime.elu, F.elu), ('selu', runtime.selu, F.selu)]
        if args.operators:ops = [op for op in ops if op[0] in args.operators]
        for dtype in (torch.float16, torch.float32):
            for n in args.sizes:
                x = torch.randn(n, dtype=dtype)*3
                for name, fast, reference in ops:
                    result, expected = fast(x), reference(x)
                    diff = result.float()-expected.float()
                    error = dict(max_abs=diff.abs().max().item(),
                                 relative_l2=(diff.norm()/expected.float().norm()).item())
                    for _ in range(4):fast(x); reference(x)
                    samples = {'tfllm': [], 'torch': []}
                    for turn in range(args.rounds):
                        order = [('torch', reference), ('tfllm', fast)]
                        if turn % 2:order.reverse()
                        for label, fn in order:
                            start = time.perf_counter()
                            for _ in range(args.repeat):result = fn(x)
                            samples[label].append((time.perf_counter()-start)*1000/args.repeat)
                    timing = {label: statistics.median(values) for label, values in samples.items()}
                    row = dict(dtype=str(dtype), n=n, op=name, **error, **timing,
                               speedup=timing['torch']/timing['tfllm'], samples=samples)
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k!='samples'}), flush=True)
    report = dict(torch=torch.__version__, backend=args.backend, threads=args.threads, repeat=args.repeat, rounds=args.rounds,
                  library=os.environ.get('TFLLM_TORCH_LIBRARY'), rows=rows)
    if args.output:args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
