#!/usr/bin/env python3
"""Dependency-light tests for persistent-runtime host orchestration."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import persistent_runtime as runtime


class _Tensor:
    dtype = "UINT8"

    def __init__(self) -> None:
        self.value: np.ndarray | None = None

    def fromNumpy(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)


class _Output:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def toNumpy(self) -> np.ndarray:
        return self.value


class _Executor:
    def __init__(self) -> None:
        self.inputs = [_Tensor(), _Tensor(), _Tensor()]

    def GetInputs(self) -> list[_Tensor]:
        return self.inputs

    def __call__(self) -> list[_Output]:
        time.sleep(0.04)
        index = int(self.inputs[0].value.reshape(-1)[0])
        return [_Output(np.asarray([index, index + 0.5], dtype=np.float32))]


def main() -> None:
    vision_options = runtime.mage.vision_executor_config(
        True,
        base={"optimize": {"MakeUnfold": True}},
    )
    assert vision_options["Core"] == [-1]
    assert vision_options["cpuLimit"] == 16
    assert vision_options["useCache"] is True
    assert vision_options["optimize"]["MakeAlign"] is True
    assert vision_options["optimize"]["AttnSoftmaxImpl"] is True
    assert vision_options["optimize"]["MakeUnfold"] is True

    prefill_options = runtime.prefill.prefill_executor_config(True)
    assert prefill_options["Core"] == [0, 1, 2, 3]
    assert prefill_options["cpuLimit"] == 16
    assert prefill_options["useCache"] is False
    assert prefill_options["optimize"] == {
        "MakeAlign": True,
        "AttnSoftmaxImpl": True,
    }

    with tempfile.TemporaryDirectory(prefix="mage-vision-pool-") as directory:
        root = Path(directory)
        manifest = {
            "canvas_height": 2,
            "canvas_width": 2,
            "canvases": [
                {"index": index, "patch_positions": [[0, 0, 0]] * 4}
                for index in range(4)
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        vision = runtime.PersistentVisionRuntime.__new__(
            runtime.PersistentVisionRuntime
        )
        vision.model_path = root
        vision.workers = 4
        vision.executors = [_Executor() for _ in range(4)]
        old_config = runtime.mage.MageVisionConfig.from_model
        old_inputs = runtime.vision_export.canvas_inputs
        runtime.mage.MageVisionConfig.from_model = staticmethod(
            lambda *args, **kwargs: SimpleNamespace(out_hidden_size=2)
        )
        runtime.vision_export.canvas_inputs = lambda root, entry, config: (
            np.asarray([entry["index"]], dtype=np.uint8),
            None,
            np.zeros((1,), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
        )
        try:
            result = vision.run(root)
        finally:
            runtime.mage.MageVisionConfig.from_model = old_config
            runtime.vision_export.canvas_inputs = old_inputs
        values = np.fromfile(root / "visual_embeddings.f32", dtype=np.float32)
        assert values.reshape(4, 2)[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert result["workers"] == 4
        assert result["execute_seconds"] < result["sum_canvas_execute_seconds"]
    print("Mage-Vit persistent runtime tests: OK")


if __name__ == "__main__":
    main()
