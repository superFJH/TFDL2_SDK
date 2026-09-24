"""Offline automatic-adaptation smoke/benchmark sample; no model downloads.

Use --kind llama/diffusers for optimize(), functional for the compile backend.
Only a production NPU library measures board performance, not an injected SDK.
"""
import argparse
import json
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from tfllm.torch import NpuRuntime, make_backend, optimize, optimization_report, restore


class FunctionalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 768)
        self.out = nn.Linear(256, 256)

    def forward(self, x):
        q, k, v = self.qkv(x).view(x.shape[0], x.shape[1], 3, 4, 64).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(y.transpose(1, 2).reshape_as(x)) + x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('llama', 'diffusers', 'functional'), default='llama')
    parser.add_argument('--model', help='optional local Transformers checkpoint directory; never downloads')
    parser.add_argument('--backend', choices=('npu', 'cpu'), default='npu')
    parser.add_argument('--tokens', type=int, default=257)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--fuse-workers', type=int, default=16)
    args = parser.parse_args()
    if min(args.tokens, args.batch, args.repeat) <= 0 or args.warmup < 0:
        parser.error('positive tokens/batch/repeat and nonnegative warmup required')
    if args.model and args.kind != 'llama':
        parser.error('--model is for the Transformers case only')
    torch.manual_seed(41)
    if args.kind == 'llama':
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
        if args.model:
            model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                dtype=torch.float16, attn_implementation='sdpa').cpu().eval()
        else:
            model = LlamaForCausalLM(LlamaConfig(vocab_size=1024, hidden_size=256, intermediate_size=768,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=max(4096, args.tokens+8), attn_implementation='sdpa')).half().eval()
        inputs = torch.randint(3, model.config.vocab_size, (args.batch, args.tokens))
        def forward(target):
            return target(inputs, use_cache=False).logits
    elif args.kind == 'diffusers':
        from diffusers.models.attention_processor import Attention
        model = Attention(query_dim=256, heads=4, dim_head=64).half().eval()
        inputs = torch.randn(args.batch, args.tokens, 256, dtype=torch.float16) * .2
        def forward(target):
            return target(inputs)
    else:
        model = FunctionalBlock().half().eval()
        inputs = torch.randn(args.batch, args.tokens, 256, dtype=torch.float16) * .2
        def forward(target):
            return target(inputs)

    def measure(target, label):
        times = []
        with torch.inference_mode():
            for _ in range(args.warmup):
                forward(target)
            for i in range(args.repeat):
                start = time.perf_counter()
                result = forward(target)
                ms = (time.perf_counter() - start)*1000
                times.append(ms)
                print(f'AUTO_RUN variant={label} run={i} ms={ms:.6f}', flush=True)
        return result, statistics.median(times)

    print(f'AUTO_CONFIG kind={args.kind} backend={args.backend} tokens={args.tokens} batch={args.batch}', flush=True)
    reference, before = measure(model, 'original')
    with NpuRuntime(backend=args.backend, attention_fuse_workers=args.fuse_workers) as runtime:
        start = time.perf_counter()
        if args.kind == 'functional':
            backend = make_backend(runtime=runtime, strict=True)
            accelerated = torch.compile(model, backend=backend, fullgraph=True)
            report = backend.report
        else:
            accelerated = optimize(model, runtime=runtime, strict=True)
            report = optimization_report(model)
        print(f'AUTO_SETUP ms={(time.perf_counter()-start)*1000:.6f} compile_is_lazy={args.kind == "functional"}', flush=True)
        actual, after = measure(accelerated, 'tfllm')
        error = (actual.float() - reference.float()).abs()
        print(f'AUTO_RESULT original_ms={before:.6f} tfllm_ms={after:.6f} ratio={before/after:.4f} '
              f'max_abs={error.max().item():.8f} mean_abs={error.mean().item():.8f}', flush=True)
        print('AUTO_COVERAGE ' + json.dumps(report.to_dict(), ensure_ascii=False), flush=True)
        if args.kind == 'functional':
            backend.close()
        else:
            restore(model)
        if not torch.isfinite(actual).all():
            raise RuntimeError('non-finite optimized output')
    print('Latency is meaningful only on real hardware; validate full-model quality separately.')


if __name__ == '__main__':
    main()
