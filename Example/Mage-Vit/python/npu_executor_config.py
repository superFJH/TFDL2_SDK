#!/usr/bin/env python3
"""Shared TFExecutor options for Mage-Vit hardware execution.

Keep the two deployed workloads explicit: vision executors may use the SDK
cache and automatic core selection, while the 36 resident Qwen prefill
executors disable that cache and expose all four NPU cores.  Software-only
conversion and operator tests intentionally do not inherit hardware options.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


NPU_CPU_LIMIT = 16
VISION_NPU_CORES = (-1,)
PREFILL_NPU_CORES = (0, 1, 2, 3)


def _executor_config(
    use_hardware: bool,
    *,
    cores: tuple[int, ...],
    use_cache: bool,
    frugal_mode: bool,
    software_attn_softmax_impl: bool,
    base: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config: dict[str, Any] = deepcopy(dict(base or {}))
    config["UseHardware"] = bool(use_hardware)
    # Hardware execution follows the deployment contract unconditionally;
    # callers may disable FrugalMode only for software tensor inspection.
    config["FrugalMode"] = True if use_hardware else bool(frugal_mode)
    optimize = deepcopy(dict(config.get("optimize", {})))
    if use_hardware:
        config["Core"] = [int(core) for core in cores]
        config["cpuLimit"] = NPU_CPU_LIMIT
        config["useCache"] = bool(use_cache)
        optimize["MakeAlign"] = True
        optimize["AttnSoftmaxImpl"] = True
    else:
        optimize["AttnSoftmaxImpl"] = bool(
            software_attn_softmax_impl
        )
    config["optimize"] = optimize
    return config


def vision_executor_config(
    use_hardware: bool,
    *,
    frugal_mode: bool = True,
    software_attn_softmax_impl: bool = True,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared-config/automatic-core vision Executor options."""
    return _executor_config(
        use_hardware,
        cores=VISION_NPU_CORES,
        use_cache=True,
        frugal_mode=frugal_mode,
        software_attn_softmax_impl=software_attn_softmax_impl,
        base=base,
    )


def prefill_executor_config(
    use_hardware: bool,
    *,
    frugal_mode: bool = True,
    software_attn_softmax_impl: bool = False,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the no-cache/all-core options for resident prefill layers."""
    return _executor_config(
        use_hardware,
        cores=PREFILL_NPU_CORES,
        use_cache=False,
        frugal_mode=frugal_mode,
        software_attn_softmax_impl=software_attn_softmax_impl,
        base=base,
    )
