#!/usr/bin/env python3
"""Inspect the local GELab checkpoint without loading its 4B parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_compat import prepend_shared_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    prepend_shared_paths()
    from checkpoint import ModelContract, SafeTensorIndex

    root = Path(args.model_path).resolve()
    contract = ModelContract.from_model(root)
    index = SafeTensorIndex(root)
    config = json.loads((root / "config.json").read_text())
    report = {
        "ok": True,
        "model_path": str(root),
        "architecture": config.get("architectures", []),
        "dtype": config.get("dtype"),
        "contract": contract.as_dict(),
        "parameter_tensor_count": len(index.weight_map),
        "tied_lm_head": "lm_head.weight" not in index.weight_map,
        "checkpoint_files": sorted(set(index.weight_map.values())),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

