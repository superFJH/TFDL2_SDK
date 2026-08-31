#!/usr/bin/env python3
"""Cheap checkpoint/config contract test; no 4B model load is performed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
from project_compat import prepend_shared_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    prepend_shared_paths()
    from checkpoint import ModelContract, SafeTensorIndex, embedding_weight_name

    contract = ModelContract.from_model(args.model_path)
    index = SafeTensorIndex(args.model_path)
    assert contract.hidden_size == 2560
    assert contract.query_size == 4096
    assert contract.num_hidden_layers == 36
    assert contract.vision_depth == 24
    assert contract.deepstack_visual_indexes == (5, 11, 17)
    assert embedding_weight_name(index) in index.weight_map
    config = json.loads((PROJECT_ROOT / "deploy/deployment.json").read_text())
    assert config["runtime_assets_dir"] == "model/runtime"
    assert config["tfdl"]["vision"]["grid_h"] * config["tfdl"]["vision"]["grid_w"] == 576
    assert config["tfdl"]["prefill"]["seq_len"] == 256
    vision_executor = config["tfdl"]["vision"]["executor"]
    prefill_executor = config["tfdl"]["prefill"]["executor"]
    assert vision_executor["UseHardware"] is True
    assert vision_executor["Core"] == list(range(8))
    assert vision_executor["cpuLimit"] == 32
    assert vision_executor["useCache"] is True
    assert prefill_executor["UseHardware"] is True
    assert prefill_executor["Core"] == list(range(8))
    assert prefill_executor["cpuLimit"] == 32
    assert prefill_executor["useCache"] is True
    for executor in (vision_executor, prefill_executor):
        assert executor["optimize"]["MakeAlign"] is True
        assert executor["optimize"]["MakeUnfold"] is True
        assert executor["optimize"]["RuntimeQuantMatMul"] is True
    assert config["decode"]["engine"] == "llama.cpp-external-kv"
    assert config["decode"]["precision"] == "Q4_K_M"
    assert config["decode"]["threads"] == 32
    assert "attention_mask.sum()" in config["decode"]["valid_kv_length"]
    assert config["decode"]["cache_dtype"] == "float16"
    print(json.dumps({"ok": True, "contract": contract.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
