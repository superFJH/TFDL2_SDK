#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mage_vit as MODULE  # noqa: E402


def make_config():
    return MODULE.MageVisionConfig(
        canvas_height=64,
        canvas_width=64,
        patch_size=16,
        spatial_merge_size=2,
        hidden_size=64,
        intermediate_size=128,
        num_layers=1,
        num_heads=1,
        out_hidden_size=32,
        layer_norm_eps=1e-6,
        rope_theta=10000.0,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        rescale_factor=1.0 / 255.0,
    )


def fake_weights(config):
    rng = np.random.default_rng(7)

    def weight(shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    result = {
        "patch.weight": weight((64, 3, 16, 16)),
        "patch.bias": np.zeros(64, np.float32),
        "layernorm_pre.weight": np.ones(64, np.float32),
        "layernorm_pre.bias": np.zeros(64, np.float32),
        "merger.norm.weight": np.ones(64, np.float32),
        "merger.norm.bias": np.zeros(64, np.float32),
        "merger.fc1.weight": weight((256, 256, 1, 1)),
        "merger.fc1.bias": np.zeros(256, np.float32),
        "merger.fc2.weight": weight((32, 256, 1, 1)),
        "merger.fc2.bias": np.zeros(32, np.float32),
    }
    p = "layers.0."
    result.update(
        {
            p + "norm1.weight": np.ones(64, np.float32),
            p + "norm1.bias": np.zeros(64, np.float32),
            p + "qkv.weight": weight((192, 64, 1, 1)),
            p + "qkv.bias": np.zeros(192, np.float32),
            p + "proj.weight": weight((64, 64, 1, 1)),
            p + "proj.bias": np.zeros(64, np.float32),
            p + "norm2.weight": np.ones(64, np.float32),
            p + "norm2.bias": np.zeros(64, np.float32),
            p + "fc1.weight": weight((128, 64, 1, 1)),
            p + "fc1.bias": np.zeros(128, np.float32),
            p + "fc2.weight": weight((64, 128, 1, 1)),
            p + "fc2.bias": np.zeros(64, np.float32),
        }
    )
    return result


def test_layout_and_rope(config) -> None:
    config.validate()
    positions = MODULE.block_order_positions(config)
    assert positions.shape == (16, 3)
    assert positions[:4].tolist() == [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]
    ]
    sin, cos = MODULE.compute_rope(positions, head_dim=64)
    assert tuple(sin.shape) == (1, 1, 16, 64)
    assert tuple(cos.shape) == (1, 1, 16, 64)
    assert torch.allclose(sin[0, 0, 0], torch.zeros(64))
    assert torch.allclose(cos[0, 0, 0], torch.ones(64))
    assert torch.allclose(sin[..., :32], sin[..., 32:])
    assert torch.allclose(cos[..., :32], cos[..., 32:])

    row_major = torch.arange(16).view(1, 1, 4, 4)
    reordered = MODULE._reorder_patch_tokens(row_major, config)
    assert reordered.flatten().tolist() == [
        0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15
    ]


def test_tfdl_float_and_quant(config) -> None:
    from TFDL2 import TFContext, TFExecutor
    from TFDL2.utils import LoadCustomOp

    addon = Path(__file__).resolve().parents[3] / "AddonOps/build/libTFDLAddOn.so"
    LoadCustomOp(str(addon))
    weights = fake_weights(config)
    positions = MODULE.block_order_positions(config)
    sin, cos = MODULE.compute_rope(positions, head_dim=64)
    rng = np.random.default_rng(9)
    raw_uint8 = rng.integers(
        0, 256, size=(1, 3, 64, 64), dtype=np.uint8
    )
    torch_graph = MODULE.TorchMageVision(
        config, weights, torch.device("cpu")
    )
    reference = torch_graph(
        torch.from_numpy(raw_uint8), torch.from_numpy(positions)
    ).numpy()

    context, inputs, outputs, _ = MODULE.build_tfdl_graph(config, weights)
    executor = TFExecutor(
        context,
        {
            "UseHardware": False,
            "FrugalMode": True,
            "optimize": {"AttnSoftmaxImpl": True},
        },
    )
    tensors = executor.GetInputs()
    tensors[0].fromNumpy(raw_uint8.astype(np.float32))
    tensors[1].fromNumpy(sin.numpy())
    tensors[2].fromNumpy(cos.numpy())
    got = executor()[0].toNumpy()
    assert np.max(np.abs(reference - got)) < 1e-5

    with tempfile.TemporaryDirectory(prefix="megavit_test_") as temporary:
        ranges = Path(temporary) / "ranges.json"
        quant = Path(temporary) / "tiny.quant.fb"
        torch_graph.collector.dump(ranges)
        qcontext, qinputs, qoutputs, qsymbols = MODULE.build_tfdl_graph(
            config, weights, range_json=ranges
        )
        MODULE.quantize_graph(
            qcontext, qinputs, qoutputs, quant,
            symbols=qsymbols, profile="mixed",
        )
        quant_loaded_context = TFContext(path=str(quant))
        quant_executor = TFExecutor(
            quant_loaded_context,
            {
                "UseHardware": False,
                "FrugalMode": True,
                "optimize": {"AttnSoftmaxImpl": True},
            },
        )
        tensors = quant_executor.GetInputs()
        tensors[0].fromNumpy(raw_uint8)
        tensors[1].fromNumpy(sin.numpy())
        tensors[2].fromNumpy(cos.numpy())
        quantized = quant_executor()[0].toNumpy().astype(np.float32)
        cosine = float(
            np.dot(reference.ravel(), quantized.ravel())
            / (np.linalg.norm(reference) * np.linalg.norm(quantized))
        )
        # This is a runtime/quantization smoke test over an untrained random
        # attention block, not an accuracy threshold. Real acceptance uses
        # representative codec canvases and the joint Mage-VL checkpoint.
        assert np.isfinite(quantized).all()
        assert cosine > 0.10, cosine

        fp16_quant = Path(temporary) / "tiny.int8_fp16_topk.quant.fb"
        bypass_report = MODULE.select_outlier_branches(config, ranges, 1)
        fp16_context, fp16_inputs, fp16_outputs, fp16_symbols = (
            MODULE.build_tfdl_graph(
                config,
                weights,
                range_json=ranges,
                explicit_qdq=True,
                fp_attn_layers=bypass_report["fp_attn_layers"],
                fp_mlp_layers=bypass_report["fp_mlp_layers"],
                per_channel_qk=True,
                per_channel_qk_max_requant_multiplier=0.99,
            )
        )
        report = MODULE.quantize_graph(
            fp16_context,
            fp16_inputs,
            fp16_outputs,
            fp16_quant,
            symbols=fp16_symbols,
            profile="int8-fp16-topk",
            config=config,
            weights=weights,
            range_json=ranges,
            outlier_top_k=1,
            per_channel_qk=True,
            bypass_report=bypass_report,
        )
        assert report is not None
        assert len(report["selected"]) == 1
        fp16_loaded_context = TFContext(path=str(fp16_quant))
        fp16_executor = TFExecutor(
            fp16_loaded_context,
            {
                "UseHardware": False,
                # The H*S contract checks below inspect internal QK/Requant
                # tensors, which are unavailable in frugal execution mode.
                "FrugalMode": False,
                "optimize": {"AttnSoftmaxImpl": True},
            },
        )
        tensors = fp16_executor.GetInputs()
        tensors[0].fromNumpy(
            raw_uint8.astype(np.float32)
            if "FLOAT" in str(tensors[0].dtype)
            else raw_uint8
        )
        tensors[1].fromNumpy(sin.numpy())
        tensors[2].fromNumpy(cos.numpy())
        fp16_output = fp16_executor()[0].toNumpy()
        assert fp16_output.dtype == np.float16
        assert np.isfinite(fp16_output).all()
        assert "FLOAT" in str(tensors[0].dtype)
        assert "FLOAT16" in str(
            fp16_executor.GetTensorByName(
                fp16_symbols["patch.tokens"]
            ).dtype
        )
        assert "UINT8" in str(
            fp16_executor.GetTensorByName(
                fp16_symbols["layers.0.norm1.quantized"]
            ).dtype
        )
        assert "UINT8" in str(
            fp16_executor.GetTensorByName(
                fp16_symbols["layers.0.qkv"]
            ).dtype
        )
        for tag in (
            "layers.0.resid1",
            "layers.0.resid2",
            "merger.norm",
            "merger.fc2",
            "output.embeddings",
        ):
            assert "FLOAT16" in str(
                fp16_executor.GetTensorByName(fp16_symbols[tag]).dtype
            ), tag
        assert not any(
            name.startswith(("PostQuant_", "PostDeQuant_"))
            for name in fp16_loaded_context.GetAllTensorNames()
        )
        q_tensor = fp16_executor.GetTensorByName(
            fp16_symbols["layers.0.q_matmul_input"]
        )
        k_tensor = fp16_executor.GetTensorByName(
            fp16_symbols["layers.0.k_matmul_input"]
        )
        qk_tensor = fp16_executor.GetTensorByName(
            fp16_symbols["layers.0.qk_matmul"]
        )
        score_tensor = fp16_executor.GetTensorByName(
            fp16_symbols["layers.0.attn_scores"]
        )
        qk_scales = np.asarray(qk_tensor.qscale, dtype=np.float64)
        multiplier = (
            float(q_tensor.qscale[0])
            * float(k_tensor.qscale[0])
            / qk_scales
        )
        assert len(qk_tensor.qmin) == config.num_heads * config.seq_len
        assert float(multiplier.max()) < 1.0
        assert np.array_equal(qk_tensor.toNumpy(), score_tensor.toNumpy())
        print(f"tiny TFDL float max_abs={np.max(np.abs(reference-got)):.3g}")
        print(f"tiny TFDL INT8 cosine={cosine:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tfdl", action="store_true")
    args = parser.parse_args()
    config = make_config()
    test_layout_and_rope(config)
    if args.with_tfdl:
        test_tfdl_float_and_quant(config)
    print("Mage-ViT Python layout/RoPE tests passed")


if __name__ == "__main__":
    main()
