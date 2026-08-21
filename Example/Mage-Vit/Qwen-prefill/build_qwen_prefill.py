#!/usr/bin/env python3
"""Build one fixed-sequence Mage Qwen3 prefill layer for TFDL/NPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qwen_prefill as prefill


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", required=True)
    result.add_argument("--layer-id", type=int, required=True)
    result.add_argument("--seq-len", type=int, required=True)
    result.add_argument("--dump-fb")
    result.add_argument("--dump-quant-fb")
    result.add_argument("--range-json")
    result.add_argument("--outlier-top-k", type=int, default=2)
    result.add_argument(
        "--quant-mode",
        choices=("quantlite", "manual", "sdk-ptq"),
        default="quantlite",
        help=(
            "quantlite builds explicit Q/DQ without Modify; manual/sdk-ptq "
            "are retained as diagnostics"
        ),
    )
    result.add_argument(
        "--attention-mode",
        choices=("arm-causal-hxs", "arm-causal-scalar", "legacy-fp16"),
        default="arm-causal-hxs",
        help=(
            "arm-causal-hxs uses H*S QK, identity Requant and the ARM "
            "causal Softmax custom op; legacy-fp16 retains mask/Add/Softmax"
        ),
    )
    result.add_argument(
        "--activation-granularity",
        choices=("scalar", "token"),
        default="scalar",
        help=(
            "token is a diagnostic MatMul experiment that is incompatible "
            "with per-output-channel weights in the current SDK; scalar "
            "uses the production Vit-style Conv1x1 path"
        ),
    )
    result.add_argument(
        "--per-channel-qk-max-requant-multiplier",
        type=float,
        default=0.99,
    )
    result.add_argument(
        "--softmax-threads",
        type=int,
        default=0,
        help="ARM Softmax OpenMP threads; 0 uses the runtime default",
    )
    result.add_argument(
        "--token-group-boundaries",
        type=int,
        nargs="*",
        default=None,
        metavar="TOKEN",
        help=(
            "Split the non-Top-K MLP into contiguous Tok-hybrid branches at "
            "these token offsets. For the soccer S=898 prompt use 21 887 "
            "(prefix / visual context / final question)."
        ),
    )
    result.add_argument(
        "--prompt-dir",
        help=(
            "Prepared prompt used to infer prefix/visual/final-query Tok "
            "hybrid boundaries; mutually exclusive with explicit boundaries."
        ),
    )
    result.add_argument(
        "--token-hybrid-qkv-start-layer",
        type=int,
        default=None,
        help=(
            "Optionally apply the same FP16-split Tok hybrid to Q/K/V from "
            "this decoder layer onward; MLP splitting is independent."
        ),
    )
    result.add_argument("--dump-symbol-map")
    result.add_argument("--dump-modify-json")
    result.add_argument("--dump-bypass-report")
    result.add_argument(
        "--debug-output",
        action="append",
        default=None,
        help=(
            "Replace quantized model outputs with logical tensor names for "
            "stage isolation, for example self_attn.qk_matmul"
        ),
    )
    result.add_argument(
        "--addon-path",
        default=str(
            Path(__file__).resolve().parents[3]
            / "AddonOps/build/libTFDLAddOn.so"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if not args.dump_fb and not args.dump_quant_fb:
        raise ValueError("pass --dump-fb and/or --dump-quant-fb")
    if args.dump_quant_fb and not args.range_json:
        raise ValueError("--dump-quant-fb requires --range-json")
    config = prefill.QwenPrefillConfig.from_model(args.model_path)
    if args.prompt_dir and args.token_group_boundaries is not None:
        raise ValueError(
            "--prompt-dir and --token-group-boundaries are mutually exclusive"
        )
    token_group_boundaries = (
        prefill.infer_token_group_boundaries(args.prompt_dir, args.seq_len)
        if args.prompt_dir
        else tuple(args.token_group_boundaries or ())
    )
    weights = prefill.load_layer_weights(args.model_path, args.layer_id)
    context = None
    inputs: list[str] = []
    outputs: list[str] = []
    symbols: dict[str, str] = {}
    need_float_context = bool(
        args.dump_fb
        or (args.dump_quant_fb and args.quant_mode != "quantlite")
    )
    if need_float_context:
        context, _, inputs, outputs, symbols = prefill.build_layer_graph(
            config,
            args.layer_id,
            args.seq_len,
            weights,
            range_json=args.range_json,
            addon_path=args.addon_path,
            fp16_boundaries=bool(
                args.dump_quant_fb and args.quant_mode == "manual"
            ),
            export_kv=not bool(
                args.dump_quant_fb and args.quant_mode == "manual"
            ),
        )
    if args.dump_fb:
        assert context is not None
        prefill.dump_context(context, args.dump_fb)
    report = None
    if args.dump_quant_fb:
        if args.quant_mode == "quantlite":
            context, _, inputs, outputs, report = (
                prefill.build_quantlite_int8_layer_graph(
                    config,
                    args.layer_id,
                    args.seq_len,
                    weights,
                    args.range_json,
                    top_k=args.outlier_top_k,
                    attention_mode=args.attention_mode,
                    activation_granularity=args.activation_granularity,
                    per_channel_qk_max_requant_multiplier=(
                        args.per_channel_qk_max_requant_multiplier
                    ),
                    softmax_threads=args.softmax_threads,
                    token_group_boundaries=token_group_boundaries,
                    token_hybrid_qkv_start_layer=(
                        args.token_hybrid_qkv_start_layer
                    ),
                    debug_outputs=args.debug_output,
                    addon_path=args.addon_path,
                )
            )
            symbols = dict(report["symbols"])
            if args.debug_output:
                outputs = list(report["debug_outputs"])
                context.SetOutputs(outputs)
            prefill.dump_context(context, args.dump_quant_fb)
        elif args.quant_mode == "manual":
            assert context is not None
            outputs, report = prefill.manually_quantize_float_layer_graph(
                context,
                config,
                args.layer_id,
                weights,
                symbols,
                args.range_json,
                top_k=args.outlier_top_k,
                dump_modify_json=args.dump_modify_json,
                dump_report=args.dump_bypass_report,
            )
            prefill.dump_context(context, args.dump_quant_fb)
        else:
            assert context is not None
            report = prefill.quantize_layer_graph(
                context,
                config,
                args.layer_id,
                weights,
                inputs,
                outputs,
                symbols,
                args.range_json,
                args.dump_quant_fb,
                top_k=args.outlier_top_k,
                dump_modify_json=args.dump_modify_json,
                dump_bypass_report=args.dump_bypass_report,
            )
    if args.dump_symbol_map:
        Path(args.dump_symbol_map).write_text(
            json.dumps(symbols, indent=2, sort_keys=True)
        )
    print(
        json.dumps(
            {
                "float_artifact": args.dump_fb,
                "quant_artifact": args.dump_quant_fb,
                "quant_mode": args.quant_mode if args.dump_quant_fb else None,
                "attention_mode": (
                    args.attention_mode if args.dump_quant_fb else None
                ),
                "layer": args.layer_id,
                "seq_len": args.seq_len,
                "hidden_size": config.hidden_size,
                "kv_shape": [
                    1,
                    config.num_key_value_heads,
                    args.seq_len,
                    config.head_dim,
                ],
                "inputs": inputs,
                "outputs": outputs,
                "symbols": len(symbols),
                "quantization": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
