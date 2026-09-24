"""Model-scoped inference adaptation. No global Torch monkey-patching."""
import collections
import threading
import types
import warnings
import weakref

import torch
from torch import nn

from .modules import NpuLinear, NpuConv2d


class Unsupported(RuntimeError):
    """A known coverage boundary, never a device execution failure."""


class OptimizationReport:
    def __init__(self):
        self.entries = []
        self.scope = "recognized modules/integrations only; functional/custom code is not audited"
        self._calls = collections.Counter()
        self._lock = threading.Lock()

    def add(self, path, operator, status, reason=""):
        self.entries.append(dict(path=path or "<root>", operator=operator, status=status, reason=reason))

    def record(self, path, status, reason=""):
        with self._lock:
            self._calls[(path or "<root>", status, reason)] += 1

    def to_dict(self):
        with self._lock:
            calls = [dict(path=p, status=s, reason=r, count=n) for (p, s, r), n in self._calls.items()]
        return {"scope": self.scope,
                "entries": [dict(e) for e in self.entries], "calls": calls}

    def __str__(self):
        lines = [f"TFLLM coverage ({self.scope})"]
        lines += [f"{e['status']:14s} {e['path']}: {e['operator']} {e['reason']}" for e in self.entries]
        with self._lock:
            lines += [f"calls={n} {p}: {s} {r}" for (p, s, r), n in self._calls.items()]
        return "\n".join(lines)


def _weight_version(weight):
    if weight.device.type != "cpu" or weight.dtype != torch.float16:
        raise Unsupported("weights must be CPU float16; no implicit dtype/device conversion")
    try:
        version = weight._version
    except RuntimeError as error:
        raise Unsupported("weights need a version counter; load them outside inference_mode") from error
    return (weight.data_ptr(), version, tuple(weight.shape), tuple(weight.stride()))


class WeightCache:
    """Immutable packed snapshots keyed by tensor identity AND version/geometry."""
    def __init__(self, runtime):
        self.runtime, self.entries = runtime, {}
        self.lock = threading.RLock()

    def get(self, weight, kind, geometry=()):
        key = (id(weight), kind, geometry)
        with self.lock:
            version = _weight_version(weight)
            entry = self.entries.get(key)
            if entry is not None and entry[0]() is weight and entry[1] == version:
                return entry[2]
            if kind == "linear":
                packed = NpuLinear(weight, runtime=self.runtime)
            else:
                stride, padding, dilation, groups = geometry
                packed = NpuConv2d(weight, stride=stride, padding=padding, dilation=dilation,
                                   groups=groups, runtime=self.runtime)
            def retire(ref):
                with self.lock:
                    current = self.entries.get(key)
                    if current is not None and current[0] is ref:
                        self.entries.pop(key)
            self.entries[key] = (weakref.ref(weight, retire), version, packed)
            return packed

    def clear(self):
        with self.lock:
            self.entries.clear()


class _State:
    def __init__(self, runtime, strict):
        self.runtime, self.strict = runtime, strict
        self.report, self.weights = OptimizationReport(), WeightCache(runtime)
        self.undo = []

    def fallback(self, path, reason, original, *args, **kwargs):
        self.report.record(path, "rejected" if self.strict else "fallback", reason)
        if self.strict:
            raise Unsupported(f"{path or '<root>'}: {reason}")
        warnings.warn(f"TFLLM {path or '<root>'}: {reason}; using original implementation",
                      RuntimeWarning, stacklevel=3)
        return original(*args, **kwargs)

    def restore(self):
        for undo in reversed(self.undo):
            undo()
        self.undo.clear()
        self.weights.clear()


def _module_geometry(module):
    if type(module) is nn.Linear:
        return "linear", ()
    if type(module) is nn.Conv2d:
        if module.padding_mode != "zeros" or isinstance(module.padding, str):
            raise Unsupported("Conv2d requires numeric zero padding")
        return "conv2d", (module.stride, module.padding, module.dilation, module.groups)
    raise Unsupported("custom Linear/Conv2d subclasses require an explicit adapter")


def _patch_projection(state, module, path, kind, geometry):
    original = module.forward
    def forward(this, input):
        x = input
        try:
            if this.training or torch.is_grad_enabled():
                raise Unsupported("inference requires eval() and no_grad()/inference_mode()")
            if x.device.type != "cpu" or x.dtype != torch.float16:
                raise Unsupported("activation must be CPU float16")
            if this.bias is not None and (this.bias.device.type != "cpu" or this.bias.dtype != torch.float16):
                raise Unsupported("bias must be CPU float16")
            if kind == "conv2d" and _module_geometry(this)[1] != geometry:
                raise Unsupported("Conv2d geometry changed after optimize; restore/re-optimize")
            packed = state.weights.get(this.weight, kind, geometry)
        except Unsupported as error:
            return state.fallback(path, str(error), original, x)
        # Device/allocator/quantization errors must propagate, never replay on CPU.
        y = packed(x)
        state.report.record(path, "npu" if state.runtime.backend == "npu" else "cpu_reference")
        if this.bias is None:
            return y
        return y + (this.bias if kind == "linear" else this.bias[None, :, None, None])
    module.forward = types.MethodType(forward, module)
    state.undo.append(lambda: module.__dict__.pop("forward", None))


def optimize(model, *, runtime, strict=True, operators=("linear", "conv2d", "attention"), predicate=None):
    """Adapt an eval Module in place, preserving parameters, aliases and state_dict.

    Strict coverage applies to recognized requested operators, not arbitrary
    Python/functional code. No global monkey-patches or device/dtype conversion.
    Call restore() before changing device, training, serialization or re-adapting.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("optimize expects nn.Module; pass a pipeline's transformer/unet separately")
    if model.training:
        raise ValueError("call model.eval() before optimize")
    if hasattr(model, "_tfllm_optimization"):
        raise ValueError("model already optimized; restore before re-optimizing")
    operators = set(operators)
    if not operators or not operators <= {"linear", "conv2d", "attention"}:
        raise ValueError("operators must select linear, conv2d and/or attention")
    state = _State(runtime, strict)
    planned = []
    # named_modules removes duplicate aliases: patch the shared instance once.
    modules = list(model.named_modules())
    try:
        from .integrations import plan_attention
        attention_plans, attention_ids = plan_attention(model, modules, state, operators, predicate)
        for path, module in modules:
            kind = "linear" if isinstance(module, nn.Linear) else "conv2d" if isinstance(module, nn.Conv2d) else None
            if kind not in operators:
                continue
            if predicate is not None and not predicate(path, module):
                state.report.add(path, kind, "excluded", "predicate")
                continue
            try:
                if "forward" in module.__dict__ or hasattr(module, "_hf_hook"):
                    raise Unsupported("custom/offload forward requires explicit adaptation")
                if module.training:
                    raise Unsupported("target module is in training mode")
                kind, geometry = _module_geometry(module)
                state.weights.get(module.weight, kind, geometry)  # Prepack before mutating model.
                planned.append((module, path, kind, geometry))
                state.report.add(path, kind, "npu" if runtime.backend == "npu" else "cpu_reference")
            except (Unsupported, NotImplementedError, TypeError, ValueError) as error:
                state.report.add(path, kind, "skipped", str(error))
                if strict:
                    raise Unsupported(f"{path or '<root>'}: {error}") from error
        if "attention" in operators:
            for path, module in modules:
                if ("attention" in type(module).__name__.lower() or isinstance(module, nn.MultiheadAttention)) and id(module) not in attention_ids:
                    if predicate is not None and not predicate(path, module):
                        state.report.add(path, "attention", "excluded", "predicate")
                        continue
                    reason = "unrecognized attention module; select a supported integration or explicit graph backend"
                    state.report.add(path, "attention", "skipped", reason)
                    if strict:
                        raise Unsupported(f"{path or '<root>'}: {reason}")
        for module, path, kind, geometry in planned:
            _patch_projection(state, module, path, kind, geometry)
        for install in attention_plans:
            install()
        model._tfllm_optimization = state
    except Exception:
        state.restore()
        raise
    return model


def optimization_report(model):
    return model._tfllm_optimization.report


def restore(model):
    """Restore original execution in place; caller must drain active inference."""
    state = model.__dict__.pop("_tfllm_optimization", None)
    if state is not None:
        state.restore()
    return model
