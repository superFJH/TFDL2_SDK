#!/usr/bin/env python3
"""Dependency-light export/quantize/external-KV contract test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import build_ort_qwen_decoder as builder
import ort_qwen_decoder as decoder


def _config() -> decoder.prefill.QwenPrefillConfig:
    return decoder.prefill.QwenPrefillConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=71,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        head_dim=8,
        attention_bias=False,
    )


def _weights(
    config: decoder.prefill.QwenPrefillConfig,
    layer: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    prefix = f"layers.{layer}."
    shapes = {
        "input_layernorm.weight": (config.hidden_size,),
        "post_attention_layernorm.weight": (config.hidden_size,),
        "self_attn.q_proj.weight": (
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        ),
        "self_attn.k_proj.weight": (
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        ),
        "self_attn.v_proj.weight": (
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        ),
        "self_attn.o_proj.weight": (
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        ),
        "self_attn.q_norm.weight": (config.head_dim,),
        "self_attn.k_norm.weight": (config.head_dim,),
        "mlp.gate_proj.weight": (config.intermediate_size, config.hidden_size),
        "mlp.up_proj.weight": (config.intermediate_size, config.hidden_size),
        "mlp.down_proj.weight": (config.hidden_size, config.intermediate_size),
    }
    result = {}
    for name, shape in shapes.items():
        if name.endswith("norm.weight"):
            value = np.ones(shape, dtype=np.float32)
        else:
            value = rng.normal(0.0, 0.035, shape).astype(np.float32)
        result[prefix + name] = value
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64).reshape(-1)
    right = right.astype(np.float64).reshape(-1)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def main() -> None:
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    config = _config()
    hidden = torch.from_numpy(rng.normal(0.0, 0.2, (1, 1, 32)).astype(np.float32))
    past_keys = [
        torch.from_numpy(rng.normal(0.0, 0.1, (1, 3, 2, 8)).astype(np.float16))
        for _ in range(2)
    ]
    past_values = [
        torch.from_numpy(rng.normal(0.0, 0.1, (1, 3, 2, 8)).astype(np.float16))
        for _ in range(2)
    ]
    sin_np, cos_np = decoder.prefill.compute_rope(np.asarray([3]), 8, config.rope_theta)
    sin = torch.from_numpy(sin_np)
    cos = torch.from_numpy(cos_np)
    modules = [
        decoder.Qwen3DecodeLayer(config, layer, _weights(config, layer, rng)).eval()
        for layer in range(2)
    ]
    current_keys = []
    current_values = []
    with torch.inference_mode():
        reference_hidden = hidden
        for layer, module in enumerate(modules):
            reference_hidden, key, value = module(
                reference_hidden, past_keys[layer], past_values[layer], sin, cos
            )
            current_keys.append(key.numpy())
            current_values.append(value.numpy())
        norm = rng.normal(1.0, 0.02, (32,)).astype(np.float32)
        head = rng.normal(0.0, 0.04, (config.vocab_size, 32)).astype(np.float32)
        final = decoder.Qwen3FinalHead(config, norm, head).eval()
        reference_logits = final(reference_hidden).numpy()

    with tempfile.TemporaryDirectory(prefix="mage-ort-test-") as directory:
        root = Path(directory)
        layer_dir = root / "layers"
        layer_dir.mkdir()
        quantized = []
        for layer, module in enumerate(modules):
            fp = layer_dir / f"layer_{layer:02d}.fp32.onnx"
            quant = layer_dir / f"layer_{layer:02d}.w8a8.onnx"
            builder._export_layer(module, fp, config, 3, 17)
            builder._quantize(fp, quant)
            builder._audit_quantized(quant)
            quantized.append(quant)
        final_fp = layer_dir / "final_head.fp32.onnx"
        final_quant = layer_dir / "final_head.w8a8.onnx"
        builder._export_final_head(final, final_fp, config, 17)
        builder._quantize(final_fp, final_quant)
        combined = root / "decoder.w8a8.onnx"
        builder.compose_decoder(quantized, final_quant, combined)
        standalone_head = root / "final_head.w8a8.onnx"
        builder.extract_final_head_model(combined, standalone_head, config)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(combined), sess_options=options, providers=["CPUExecutionProvider"]
        )
        assert {
            item.name for item in session.get_inputs()
        } == decoder.expected_input_names(2)
        assert {
            item.name for item in session.get_outputs()
        } == decoder.expected_output_names(2)
        feeds = {
            decoder.HIDDEN_INPUT: hidden.numpy(),
            decoder.SIN_INPUT: sin_np,
            decoder.COS_INPUT: cos_np,
        }
        for layer in range(2):
            feeds[decoder.past_key_name(layer)] = past_keys[layer].numpy()
            feeds[decoder.past_value_name(layer)] = past_values[layer].numpy()
        output_names = [item.name for item in session.get_outputs()]
        outputs = dict(zip(output_names, session.run(output_names, feeds)))
        head_session = ort.InferenceSession(
            str(standalone_head),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        head_logits = head_session.run(
            ["logits"], {"hidden": reference_hidden.numpy()}
        )[0]
        original_head_session = ort.InferenceSession(
            str(final_quant),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        original_head_logits = original_head_session.run(
            ["logits"], {"hidden": reference_hidden.numpy()}
        )[0]
        if not np.array_equal(head_logits, original_head_logits):
            raise AssertionError("extracted W8A8 head differs from source head")
        logit_cosine = _cosine(reference_logits, outputs[decoder.LOGITS_OUTPUT])
        if logit_cosine < 0.995:
            raise AssertionError(f"W8A8 logits cosine is too low: {logit_cosine}")
        for layer in range(2):
            key = outputs[decoder.present_key_name(layer)]
            value = outputs[decoder.present_value_name(layer)]
            if key.dtype != np.float16 or value.dtype != np.float16:
                raise AssertionError("present KV output must remain FP16")
            if _cosine(current_keys[layer], key) < 0.995:
                raise AssertionError(f"layer {layer} K-cache accuracy is too low")
            if _cosine(current_values[layer], value) < 0.995:
                raise AssertionError(f"layer {layer} V-cache accuracy is too low")
        print(f"ORT W8A8 tiny decoder: logits cosine={logit_cosine:.8f}")


if __name__ == "__main__":
    main()
