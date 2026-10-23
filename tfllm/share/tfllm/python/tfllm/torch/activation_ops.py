"""Shared, explicit activation metadata; no class-name or graph guessing."""
import torch
from torch import nn
from torch.nn import functional as F

MODULES = {nn.GELU: "gelu", nn.SiLU: "silu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh",
           nn.Mish: "mish", nn.Softplus: "softplus", nn.ELU: "elu", nn.SELU: "selu"}
FUNCTIONS = {"gelu": F.gelu, "silu": F.silu, "sigmoid": torch.sigmoid, "tanh": torch.tanh,
             "mish": F.mish, "softplus": F.softplus, "elu": F.elu, "selu": F.selu}
DEFAULTS = {"gelu": {"approximate": "none"}, "silu": {"inplace": False}, "sigmoid": {}, "tanh": {},
            "mish": {"inplace": False}, "softplus": {"beta": 1, "threshold": 20},
            "elu": {"alpha": 1, "inplace": False}, "selu": {"inplace": False}}
IDS = {"gelu": 0, "silu": 2, "sigmoid": 3, "tanh": 4, "mish": 5, "softplus": 6, "elu": 7, "selu": 8}
# Conservative A77 gates, including Python/C ABI and persistent-pool costs.
# Absent entries retain Torch. GELU's erf/tanh distinction is handled below.
AUTO_MIN_ELEMENTS = {
    torch.float16: {"gelu": 65536, "silu": 65536, "sigmoid": 65536, "tanh": 65536,
                    "mish": 4096, "softplus": 65536, "elu": 65536, "selu": 65536},
    torch.float32: {"tanh": 262144, "mish": 4096, "softplus": 65536, "elu": 65536, "selu": 65536},
}


def auto_native(kind, input, parameters):
    threshold = AUTO_MIN_ELEMENTS.get(input.dtype, {}).get(kind)
    if input.dtype == torch.float16 and kind == "gelu" and parameters.get("approximate", "none") == "none":
        threshold = 4096
    return threshold is not None and input.numel() >= threshold


def native_id(kind, parameters):
    """None means valid Torch parameters without a corresponding native kernel."""
    if kind not in DEFAULTS:
        raise ValueError("unknown activation: " + kind)
    extra = parameters.keys() - DEFAULTS[kind].keys()
    if extra:
        raise TypeError(f"unexpected {kind} parameters: {sorted(extra)}")
    values = {**DEFAULTS[kind], **parameters}
    if kind == "gelu":
        if values["approximate"] not in ("none", "tanh"):
            raise ValueError("GELU approximate must be none or tanh")
        return int(values["approximate"] == "tanh")
    # Parameterized tables are deliberately not cached without bounds. Keep
    # beta/threshold/alpha semantics in Torch unless they match these kernels.
    if any(type(values[k]) not in (int, float) or values[k] != v
           for k, v in DEFAULTS[kind].items() if k != "inplace"):
        return None
    return IDS[kind]


def module_parameters(module, kind):
    return {key: getattr(module, key) for key in DEFAULTS[kind]}


def module_kind(module):
    # Identify subclasses as candidates so optimize's usual strict preflight
    # can reject custom behavior rather than silently replacing it.
    return next((kind for cls, kind in MODULES.items() if isinstance(module, cls)), None)


# Functional and ATen positional arguments differ for ELU: ATen also has
# scale/input_scale, so that ATen overload remains in Torch.
GRAPH_FUNCTIONS = {}
for kind, fn in FUNCTIONS.items():
    names = tuple(DEFAULTS[kind])
    if kind == "gelu": names = ()  # keyword-only approximate
    GRAPH_FUNCTIONS[fn] = (kind, names)
    if kind != "elu":
        GRAPH_FUNCTIONS[getattr(torch.ops.aten, kind).default] = (kind, tuple(k for k in names if k != "inplace"))
for fn, kind in ((F.sigmoid, "sigmoid"), (F.tanh, "tanh")):
    GRAPH_FUNCTIONS[fn] = (kind, ())
