#!/usr/bin/env python3
"""Dependency-light fixed-bucket prompt padding tests."""

from __future__ import annotations

import json
import numpy as np
import tempfile
from pathlib import Path

from prepare_qwen_prefill_prompt import right_pad_prompt
from run_qwen_prefill_stack import (
    _token_group_compatibility,
    _valid_sequence_length,
)


def main() -> None:
    ids, mask, valid = right_pad_prompt(
        np.asarray([[11, 12, 13]], dtype=np.int64),
        np.ones((1, 3), dtype=np.int64),
        6,
        99,
    )
    assert valid == 3
    np.testing.assert_array_equal(ids, [[11, 12, 13, 99, 99, 99]])
    np.testing.assert_array_equal(mask, [[1, 1, 1, 0, 0, 0]])
    try:
        right_pad_prompt(ids[:, :3], mask[:, :3], 2, 99)
    except ValueError as error:
        assert "supports at most 2" in str(error)
    else:
        raise AssertionError("an oversized prompt was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        np.save(root / "attention_mask.npy", mask)
        assert _valid_sequence_length(
            root,
            6,
            {"model_seq_len": 6, "valid_seq_len": 3},
        ) == 3
        np.save(
            root / "input_ids.npy",
            np.asarray([[1, 99, 99, 2, 0, 0]], dtype=np.int64),
        )
        fb_dir = root / "fb"
        fb_dir.mkdir()
        (fb_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "token_group_boundaries": [1, 3],
                    "calibration": {"valid_seq_lens": [4, 6]},
                }
            )
        )
        compatibility = _token_group_compatibility(
            fb_dir,
            root,
            {"image_token_id": 99},
            4,
        )
        assert compatibility["compatible"] is True
        assert compatibility["inside_calibrated_length_span"] is True
        manifest = json.loads((fb_dir / "manifest.json").read_text())
        manifest["token_group_boundaries"] = [1]
        (fb_dir / "manifest.json").write_text(json.dumps(manifest))
        coarse = _token_group_compatibility(
            fb_dir,
            root,
            {"image_token_id": 99},
            4,
        )
        assert coarse["compatible"] is True
    print("Qwen fixed-bucket prompt padding tests: OK")


if __name__ == "__main__":
    main()
