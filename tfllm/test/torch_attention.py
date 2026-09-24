"""Compare old Torch-softmax and native fused attention on the board.

FP32 reference checks sample query rows, never a full L x S activation.
Injected SDK timings are NOT board performance; use a production NPU library.
"""
import argparse
import statistics
import time

import torch
from torch.nn import functional as F
from tfllm.torch import NpuAttention, NpuRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1972)
    parser.add_argument("--keys", type=int, default=0, help="0: same as tokens")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=0, help="0: same as heads")
    parser.add_argument("--head-dim", type=int, default=72)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--fuse-workers", type=int, default=16)
    parser.add_argument("--score-mib", type=int, default=4)
    parser.add_argument("--no-pipeline", action="store_true")
    args = parser.parse_args()
    keys, kv_heads = args.keys or args.tokens, args.kv_heads or args.heads
    if (min(args.tokens, keys, args.heads, kv_heads, args.head_dim, args.repeat) <= 0
            or args.warmup < 0 or args.heads % kv_heads):
        parser.error("positive dimensions/repeat, nonnegative warmup and Hq % Hkv == 0 required")
    torch.manual_seed(41)
    q = torch.randn(1, args.heads, args.tokens, args.head_dim, dtype=torch.float16) * .3
    k = torch.randn(1, kv_heads, keys, args.head_dim, dtype=torch.float16) * .3
    v = torch.randn_like(k) * .3
    print(f"TORCH_ATTENTION tokens={args.tokens} keys={keys} heads={args.heads} kv_heads={kv_heads} "
          f"head_dim={args.head_dim} causal={args.causal} workers={args.fuse_workers} "
          f"score_mib={args.score_mib} pipeline={not args.no_pipeline}", flush=True)
    times, outputs = {"torch": [], "fused": []}, {}
    with NpuRuntime(attention_fuse_workers=args.fuse_workers, attention_score_mib=args.score_mib,
                    attention_pipeline=not args.no_pipeline) as runtime, torch.inference_mode():
        layers = {name: NpuAttention(runtime=runtime, implementation=name) for name in times}
        def run(name):
            return layers[name](q, k, v, is_causal=args.causal, enable_gqa=True)
        for _ in range(args.warmup):
            for name in layers:
                run(name)
        for i in range(args.repeat):
            for name in (("torch", "fused") if i % 2 == 0 else ("fused", "torch")):
                start = time.perf_counter()
                y = run(name)  # Native call is synchronous; no torch device sync.
                ms = (time.perf_counter() - start) * 1000
                outputs[name] = y
                times[name].append(ms)
                print(f"TORCH_ATTENTION_RUN variant={name} run={i} ms={ms:.6f}", flush=True)
        error = (outputs["torch"].float() - outputs["fused"].float()).abs().max().item()
        # Native preserves TFLLM's FP16 score/probability boundaries. Old Torch
        # float softmax may differ slightly; this is not a bit-parity test.
        indices = torch.linspace(0, args.tokens - 1, min(8, args.tokens)).long()
        max_reference_error = 0.0
        for head in range(args.heads):
            kh = head // (args.heads // kv_heads)
            mask = torch.arange(keys)[None, :] <= indices[:, None] if args.causal else None
            ref = F.scaled_dot_product_attention(q[0, head, indices].float(), k[0, kh].float(),
                                                 v[0, kh].float(), attn_mask=mask)
            max_reference_error = max(max_reference_error, (ref - outputs["fused"][0, head, indices].float()).abs().max().item())
        baseline, fused = (statistics.median(times[name]) for name in ("torch", "fused"))
        print(f"TORCH_ATTENTION_SUMMARY baseline_ms={baseline:.6f} fused_ms={fused:.6f} "
              f"speedup={baseline / fused:.4f} old_path_max_abs={error:.6f} fp32_sample_max_abs={max_reference_error:.6f}")
        if not torch.isfinite(outputs["fused"]).all() or error > .01 or max_reference_error > .01:
            raise RuntimeError("attention accuracy check failed")
        print("PASS Torch attention; throughput requires real board/library")


if __name__ == "__main__":
    main()
