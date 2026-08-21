#!/usr/bin/env python3
"""Tiny out-of-process smoke test for the TFDL Qwen prefill quantizer."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--no-gqa", action="store_true")
    parser.add_argument("--no-kv-export", action="store_true")
    parser.add_argument("--equal-query-hidden", action="store_true")
    args = parser.parse_args()

    import torch

    head_dim = 8 if args.equal_query_hidden else 16
    kv_heads = 4 if args.no_gqa else 2
    config = prefill.QwenPrefillConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        head_dim=head_dim,
        attention_bias=False,
    )
    rng = np.random.default_rng(1234)
    prefix = "layers.0"

    def normal(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(scale=0.05, size=shape).astype(np.float32)

    weights = {
        f"{prefix}.input_layernorm.weight": np.ones(32, np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(32, np.float32),
        f"{prefix}.self_attn.q_proj.weight": normal((4 * head_dim, 32)),
        f"{prefix}.self_attn.k_proj.weight": normal((kv_heads * head_dim, 32)),
        f"{prefix}.self_attn.v_proj.weight": normal((kv_heads * head_dim, 32)),
        f"{prefix}.self_attn.o_proj.weight": normal((32, 4 * head_dim)),
        f"{prefix}.self_attn.q_norm.weight": np.ones(head_dim, np.float32),
        f"{prefix}.self_attn.k_norm.weight": np.ones(head_dim, np.float32),
        f"{prefix}.mlp.gate_proj.weight": normal((64, 32)),
        f"{prefix}.mlp.up_proj.weight": normal((64, 32)),
        f"{prefix}.mlp.down_proj.weight": normal((32, 64)),
    }
    hidden_np = normal((1, args.seq_len, 32))
    sin_np, cos_np = prefill.compute_rope(
        np.arange(args.seq_len), config.head_dim, config.rope_theta
    )
    collector = prefill.RangeCollector()
    with torch.no_grad():
        prefill.torch_layer(
            config,
            0,
            weights,
            torch.from_numpy(hidden_np),
            torch.from_numpy(sin_np),
            torch.from_numpy(cos_np),
            collector,
        )
    with tempfile.TemporaryDirectory(prefix="mage_qwen_prefill_smoke_") as temp:
        ranges = Path(temp) / "ranges.json"
        collector.dump(ranges)
        context, _, inputs, outputs, symbols = prefill.build_layer_graph(
            config,
            0,
            args.seq_len,
            weights,
            range_json=ranges,
            causal=not args.no_mask,
            qk_norm=not args.no_qk_norm,
            export_kv=not args.no_kv_export,
        )
        if args.no_mask or args.no_qk_norm or args.no_gqa or args.no_kv_export:
            from TFDL2 import CalibrationMode, TFCalibration
            from TFDL2.Common import TFDataType

            calibration = TFCalibration(
                context,
                CalibrationMode.Naive,
                {"UseHardware": False, "FrugalMode": True},
            )
            calibration.Quantize(
                {
                    inputs[0]: TFDataType.TFDL_FLOAT,
                    inputs[1]: TFDataType.TFDL_FLOAT,
                    inputs[2]: TFDataType.TFDL_FLOAT,
                },
                stopquanttensors=(outputs[0],),
                MergeConcate=False,
                Perchannel=True,
            )
            prefill.dump_context(context, args.output)
            print(f"tiny no-mask quant smoke: {args.output}")
            return
        prefill.quantize_layer_graph(
            context,
            config,
            0,
            weights,
            inputs,
            outputs,
            symbols,
            ranges,
            args.output,
            top_k=args.top_k,
        )
    print(f"tiny Qwen prefill quant smoke: {args.output}")


if __name__ == "__main__":
    main()
