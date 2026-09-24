"""Board sample: native block-CSR attention with scattered, head-specific blocks.

This synthetic mask is NOT FlashVSR's top-k/window selection algorithm.
FP32 checks sample query rows only; never allocate the full token-level mask.
Injected SDK timings are NOT board performance.
"""
import argparse
import random
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import BlockSparseMask, NpuBlockSparseAttention, NpuRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--keys', type=int, default=32768)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--kv-heads', type=int, default=2)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--value-dim', type=int, default=0, help='0: same as head-dim')
    parser.add_argument('--query-block-size', type=int, default=128)
    parser.add_argument('--kv-block-size', type=int, default=128)
    parser.add_argument('--selected-blocks', type=int, default=8)
    parser.add_argument('--empty-every', type=int, default=0, help='0: none; otherwise every Nth CSR row is empty')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--fuse-workers', type=int, default=16)
    parser.add_argument('--score-mib', type=int, default=4)
    parser.add_argument('--no-pipeline', action='store_true')
    args = parser.parse_args()
    dv = args.value_dim or args.head_dim
    if (min(args.tokens, args.keys, args.heads, args.kv_heads, args.head_dim, dv, args.repeat) <= 0
            or args.heads % args.kv_heads or args.selected_blocks < 0 or args.empty_every < 0 or args.warmup < 0
            or not 1 <= args.query_block_size <= 1024 or not 1 <= args.kv_block_size <= 16384
            or max(args.head_dim, dv) > 16384):
        parser.error('invalid dimensions, GQA ratio, block sizes or repeat/warmup counts')
    bq, bk = args.query_block_size, args.kv_block_size
    nq, nk = (args.tokens+bq-1)//bq, (args.keys+bk-1)//bk
    rng = random.Random(41)
    offsets, columns, lengths = [0], [], []
    pairs = 0
    for row in range(args.heads*nq):
        selected = [] if args.empty_every and row % args.empty_every == 0 else sorted(
            rng.sample(range(nk), min(args.selected_blocks, nk)))
        count = sum(min(bk, args.keys-block*bk) for block in selected)
        if count > 16384:
            parser.error('selected keys per query block exceed 16384; reduce --selected-blocks')
        lengths.append(count)
        pairs += count * min(bq, args.tokens-(row % nq)*bq)
        columns.extend(selected)
        offsets.append(len(columns))
    mask = BlockSparseMask(torch.tensor(offsets, dtype=torch.int64), torch.tensor(columns, dtype=torch.int64), bq, bk)
    torch.manual_seed(41)
    q = torch.randn(1, args.heads, args.tokens, args.head_dim, dtype=torch.float16) * .3
    k = torch.randn(1, args.kv_heads, args.keys, args.head_dim, dtype=torch.float16) * .3
    v = torch.randn(1, args.kv_heads, args.keys, dv, dtype=torch.float16) * .3
    dense_pairs = args.heads*args.tokens*args.keys
    print(f'BLOCK_CONFIG tokens={args.tokens} keys={args.keys} heads={args.heads} kv_heads={args.kv_heads} '
          f'dim={args.head_dim} value_dim={dv} qblock={bq} kvblock={bk} nnz={len(columns)} '
          f'selected_min={min(lengths)} selected_max={max(lengths)} density={pairs/dense_pairs:.6f} '
          f'workers={args.fuse_workers} score_mib={args.score_mib} pipeline={not args.no_pipeline}', flush=True)
    # Hypothetical full INT32 QK materialization, NOT measured traffic/peak memory.
    print(f'BLOCK_LOGICAL_QK dense_mib={dense_pairs*4/2**20:.3f} selected_mib={pairs*4/2**20:.3f} '
          'native_uses_bounded_tiles=true mask_policy=synthetic_scattered', flush=True)
    times = []
    with NpuRuntime(attention_fuse_workers=args.fuse_workers, attention_score_mib=args.score_mib,
                    attention_pipeline=not args.no_pipeline) as runtime, torch.inference_mode():
        layer = NpuBlockSparseAttention(runtime=runtime)
        for _ in range(args.warmup):
            layer(q, k, v, mask, enable_gqa=True)
        for run in range(args.repeat):
            start = time.perf_counter()
            y = layer(q, k, v, mask, enable_gqa=True)  # Synchronous native call, includes KV gather/prepare.
            ms = (time.perf_counter()-start)*1000
            times.append(ms)
            print(f'BLOCK_RUN run={run} ms={ms:.6f}', flush=True)
        max_error = 0.0
        sample_rows = torch.linspace(0, args.tokens-1, min(8, args.tokens)).long().tolist()
        for head in range(args.heads):
            kh = head // (args.heads // args.kv_heads)
            for query in sample_rows:
                row = head*nq + query//bq
                selected = columns[offsets[row]:offsets[row+1]]
                indices = torch.tensor([token for block in selected
                    for token in range(block*bk, min((block+1)*bk, args.keys))], dtype=torch.int64)
                ref = F.scaled_dot_product_attention(q[0, head, query:query+1].float(),
                    k[0, kh, indices].float(), v[0, kh, indices].float()) if indices.numel() else torch.zeros(1, dv)
                max_error = max(max_error, (ref-y[0, head, query:query+1].float()).abs().max().item())
        finite = bool(torch.isfinite(y).all())
        print(f'BLOCK_SUMMARY median_ms={statistics.median(times):.6f} min_ms={min(times):.6f} '
              f'fp32_sample_max_abs={max_error:.6f} finite={finite}', flush=True)
        if not finite or max_error > .01:
            raise RuntimeError('block attention accuracy check failed')
        print('PASS block attention; throughput requires real board/library')


if __name__ == '__main__':
    main()
