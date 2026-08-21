#!/usr/bin/env python3
"""Dependency-light tests for the Qwen prefill graph contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import qwen_prefill as prefill


def test_hardware_executor_config_contract() -> None:
    # Hardware must retain the deployment settings even when a debug caller
    # requests non-frugal software execution.
    hardware = prefill.prefill_executor_config(True, frugal_mode=False)
    assert hardware == {
        "UseHardware": True,
        "FrugalMode": True,
        "Core": [0, 1, 2, 3],
        "cpuLimit": 16,
        "useCache": False,
        "optimize": {
            "MakeAlign": True,
            "AttnSoftmaxImpl": True,
        },
    }
    software = prefill.prefill_executor_config(
        False,
        frugal_mode=False,
        software_attn_softmax_impl=False,
    )
    assert software == {
        "UseHardware": False,
        "FrugalMode": False,
        "optimize": {"AttnSoftmaxImpl": False},
    }


def test_config_reads_mage_qwen3_shape_and_rope() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "hidden_size": 40,
                        "intermediate_size": 96,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 16,
                        "vocab_size": 128,
                        "rope_parameters": {"rope_theta": 5_000_000},
                    }
                }
            )
        )
        config = prefill.QwenPrefillConfig.from_model(root)
    assert config.hidden_size == 40
    assert config.query_size == 64
    assert config.head_dim == 16
    assert config.kv_repeats == 2
    assert config.rope_theta == 5_000_000


def test_rope_and_causal_mask() -> None:
    sin, cos = prefill.compute_rope(np.arange(3), 8, 10_000.0)
    assert sin.shape == (1, 1, 3, 8)
    assert cos.shape == (1, 1, 3, 8)
    np.testing.assert_allclose(sin[:, :, 0], 0.0)
    np.testing.assert_allclose(cos[:, :, 0], 1.0)
    mask = prefill.causal_mask(3, np.float16)
    assert mask.dtype == np.float16
    np.testing.assert_array_equal(
        mask[0],
        np.array(
            [[0, -65504, -65504], [0, 0, -65504], [0, 0, 0]],
            dtype=np.float16,
        ),
    )


def test_torch_layer_supports_query_size_larger_than_hidden() -> None:
    import torch

    config = prefill.QwenPrefillConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        head_dim=8,
        attention_bias=False,
    )
    rng = np.random.default_rng(7)
    prefix = "layers.0"

    def normal(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    weights = {
        f"{prefix}.input_layernorm.weight": np.ones(8, np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(8, np.float32),
        f"{prefix}.self_attn.q_proj.weight": normal((16, 8)),
        f"{prefix}.self_attn.k_proj.weight": normal((8, 8)),
        f"{prefix}.self_attn.v_proj.weight": normal((8, 8)),
        f"{prefix}.self_attn.o_proj.weight": normal((8, 16)),
        f"{prefix}.self_attn.q_norm.weight": np.ones(8, np.float32),
        f"{prefix}.self_attn.k_norm.weight": np.ones(8, np.float32),
        f"{prefix}.mlp.gate_proj.weight": normal((16, 8)),
        f"{prefix}.mlp.up_proj.weight": normal((16, 8)),
        f"{prefix}.mlp.down_proj.weight": normal((8, 16)),
    }
    hidden = torch.from_numpy(normal((1, 3, 8)))
    sin_np, cos_np = prefill.compute_rope(np.arange(3), 8, 10_000.0)
    output, key, value = prefill.torch_layer(
        config,
        0,
        weights,
        hidden,
        torch.from_numpy(sin_np),
        torch.from_numpy(cos_np),
    )
    assert output.shape == (1, 3, 8)
    assert key.shape == (1, 1, 3, 8)
    assert value.shape == (1, 1, 3, 8)


def test_range_collector_emits_token_ranges() -> None:
    collector = prefill.RangeCollector()
    value = np.array(
        [[[-2.0, 1.0, 3.0], [-7.0, -1.0, 2.0]]], dtype=np.float32
    )
    collector.observe("layers.0.input_norm", value)
    item = collector.as_json()["layers.0.input_norm.tokens"]
    assert item["channel_layout"] == "S"
    assert item["row_count"] == 2
    np.testing.assert_array_equal(item["min"], [-2.0, -7.0])
    np.testing.assert_array_equal(item["max"], [3.0, 2.0])


def test_range_collector_ignores_right_padding_and_future_qk() -> None:
    collector = prefill.RangeCollector()
    token_values = np.array(
        [[[1.0, -2.0], [3.0, -4.0], [999.0, -999.0], [888.0, -888.0]]],
        dtype=np.float32,
    )
    collector.observe(
        "layers.0.input_norm", token_values, valid_seq_len=2
    )
    qk = np.full((1, 1, 4, 4), 1000.0, dtype=np.float32)
    for query in range(4):
        qk[0, 0, query, : query + 1] = np.arange(query + 1) - 4.0
    collector.observe(
        "layers.0.self_attn.qk_matmul", qk, valid_seq_len=3
    )
    try:
        collector.as_json()
    except ValueError as error:
        assert "uncalibrated" in str(error)
    else:
        raise AssertionError("missing tail calibration rows were accepted")

    # A bucket-filling real prompt calibrates the tail. Earlier pad values
    # never enter the scalar/token unions, and future causal cells never
    # enter the HxS QK ranges.
    collector.observe(
        "layers.0.input_norm",
        np.array(
            [[[0.5, -1.0], [2.0, -3.0], [4.0, -5.0], [6.0, -7.0]]],
            dtype=np.float32,
        ),
        valid_seq_len=4,
    )
    collector.observe(
        "layers.0.self_attn.qk_matmul", qk, valid_seq_len=4
    )
    ranges = collector.as_json()
    assert ranges["layers.0.input_norm"] == {"min": -7.0, "max": 6.0}
    rows = ranges["layers.0.self_attn.qk_matmul.rows"]
    np.testing.assert_array_equal(rows["max"], [0.0, 0.0, 0.0, 0.0])
    assert max(rows["max"]) < 1000.0


def test_prompt_token_group_boundaries() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "metadata.json").write_text(
            json.dumps({"image_token_id": 99})
        )
        np.save(
            root / "input_ids.npy",
            np.asarray([1, 2, 99, 99, 3, 99, 4, 5], dtype=np.int64),
        )
        boundaries = prefill.infer_token_group_boundaries(root, 8)
    assert boundaries == (2, 6)


def test_prompt_token_group_boundaries_ignore_padding() -> None:
    ids = np.asarray([1, 99, 99, 2, 0, 99], dtype=np.int64)
    boundaries = prefill.prompt_token_group_boundaries(
        ids, 99, valid_seq_len=4
    )
    assert boundaries == (1, 3), boundaries


if __name__ == "__main__":
    test_hardware_executor_config_contract()
    test_config_reads_mage_qwen3_shape_and_rope()
    test_rope_and_causal_mask()
    test_torch_layer_supports_query_size_larger_than_hidden()
    test_range_collector_emits_token_ranges()
    test_range_collector_ignores_right_padding_and_future_qk()
    test_prompt_token_group_boundaries()
    test_prompt_token_group_boundaries_ignore_padding()
    print("qwen prefill tests: OK")
