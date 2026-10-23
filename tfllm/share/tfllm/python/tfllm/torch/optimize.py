"""Model-scoped inference adaptation. No global Torch monkey-patching."""
import collections
import threading
import types
import warnings
import weakref

import torch
from torch import nn

from .modules import NpuLinear, NpuConv2d, NpuConvTranspose2d
from .activation_ops import MODULES as ACTIVATION_MODULES, FUNCTIONS as ACTIVATION_FUNCTIONS
from .activation_ops import native_id, module_parameters, module_kind, auto_native


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


def _weight_version(weight, precision="auto"):
    allowed = (torch.float16,) if precision == "strict" else (torch.float16, torch.float32)
    if precision == "fp16": allowed += (torch.bfloat16,)
    if weight.device.type != "cpu" or weight.dtype not in allowed:
        raise Unsupported("weights must use a supported CPU dtype (auto: float16/float32)")
    try:
        version = weight._version
    except RuntimeError as error:
        raise Unsupported("weights need a version counter; load them outside inference_mode") from error
    return (weight.data_ptr(), version, weight.dtype, tuple(weight.shape), tuple(weight.stride()))


class WeightCache:
    """Immutable packed snapshots keyed by tensor identity AND version/geometry."""
    def __init__(self, runtime, precision="auto"):
        self.runtime, self.entries, self.precision = runtime, {}, precision
        self.lock = threading.RLock()

    def get(self, weight, kind, geometry=()):
        key = (id(weight), kind, geometry)
        with self.lock:
            version = _weight_version(weight, self.precision)
            entry = self.entries.get(key)
            if entry is not None and entry[0]() is weight and entry[1] == version:
                return entry[2]
            snapshot = weight
            if self.precision == "fp16" and weight.dtype == torch.bfloat16:
                with torch.inference_mode(False), torch.no_grad():
                    snapshot = weight.detach().to(torch.float16)
                if not bool(torch.isfinite(snapshot).all()):
                    raise Unsupported("weights are non-finite or overflow the FP16 operand bridge")
            if kind == "linear":
                packed = NpuLinear(snapshot, runtime=self.runtime)
            elif kind == "conv_transpose2d":
                stride, padding, output_padding, groups, dilation = geometry
                packed = NpuConvTranspose2d(snapshot, stride=stride, padding=padding, output_padding=output_padding,
                                            groups=groups, dilation=dilation, runtime=self.runtime)
            else:
                stride, padding, dilation, groups = geometry
                packed = NpuConv2d(snapshot, stride=stride, padding=padding, dilation=dilation,
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
    def __init__(self, runtime, strict, precision="auto", activation_policy="auto"):
        if activation_policy not in ("auto", "native", "torch"):
            raise ValueError("activation_policy must be auto, native or torch")
        self.runtime, self.strict, self.precision = runtime, strict, precision
        self.activation_policy = activation_policy
        self.report, self.weights = OptimizationReport(), WeightCache(runtime, precision)
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
    if type(module) is nn.ConvTranspose2d:
        return "conv_transpose2d", (module.stride, module.padding, module.output_padding, module.groups, module.dilation)
    raise Unsupported("custom Linear/Conv2d/ConvTranspose2d subclasses require an explicit adapter")


def _patch_projection(state, module, path, kind, geometry):
    original = module.forward
    def forward(this, input, output_size=None):
        x = input
        try:
            if this.training or torch.is_grad_enabled():
                raise Unsupported("inference requires eval() and no_grad()/inference_mode()")
            allowed = (torch.float16,) if state.precision == "strict" else (torch.float16, torch.float32)
            if state.precision == "fp16": allowed += (torch.bfloat16,)
            if x.device.type != "cpu" or x.dtype not in allowed:
                raise Unsupported("activation must use a supported CPU dtype (auto: float16/float32)")
            dtype = torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else x.dtype
            if dtype not in allowed: raise Unsupported("autocast output dtype is not supported; auto supports float16/float32")
            if this.bias is not None and (this.bias.device.type != "cpu" or this.bias.dtype not in allowed):
                raise Unsupported("bias must use a supported CPU floating dtype")
            if kind in ("conv2d", "conv_transpose2d") and _module_geometry(this)[1] != geometry:
                raise Unsupported("Conv2d geometry changed after optimize; restore/re-optimize")
            packed = state.weights.get(this.weight, kind, geometry)
        except Unsupported as error:
            return state.fallback(path, str(error), original, x, **({"output_size": output_size} if kind == "conv_transpose2d" else {}))
        # Device/allocator/quantization errors must propagate, never replay on CPU.
        with torch.autocast("cpu", enabled=False):
            # Only convert if explicitly requested by autocast (or the legacy
            # BF16 bridge). FP32 is never narrowed merely to enter the NPU.
            execution_dtype = torch.float16 if dtype == torch.bfloat16 else dtype
            value = x if x.dtype == execution_dtype else x.to(execution_dtype)
            y = packed(value, output_size=output_size) if kind == "conv_transpose2d" else packed(value)
            if y.dtype != dtype: y = y.to(dtype)
        state.report.record(path, "npu" if state.runtime.backend == "npu" else "cpu_reference")
        if this.bias is None:
            return y
        bias = this.bias.to(y.dtype)
        return y + (bias if kind == "linear" else bias[None, :, None, None])
    module.forward = types.MethodType(forward, module)
    state.undo.append(lambda: module.__dict__.pop("forward", None))


def _activation_capability(runtime, kind="gelu"):
    if not all(hasattr(runtime._lib, "tfllm_torch_activation_" + dtype) for dtype in ("f16", "f32")):
        raise Unsupported("rebuild/install libtfllm-torch for fast GELU/SiLU support")
    if kind not in ("gelu", "silu") and (not hasattr(runtime._lib, "tfllm_torch_activation_version")
                                        or runtime._lib.tfllm_torch_activation_version() < 2):
        raise Unsupported("rebuild/install libtfllm-torch for extended activation support")


def _activation(state, path, kind, input, *, original=None, training=False, **parameters):
    if original is None:
        original = lambda: ACTIVATION_FUNCTIONS[kind](input, **parameters)
    try:
        if training or torch.is_grad_enabled():
            raise Unsupported("fast activations require eval() and no_grad()/inference_mode()")
        allowed = (torch.float16,) if state.precision == "strict" else (torch.float16, torch.float32)
        if input.device.type != "cpu" or input.dtype not in allowed:
            raise Unsupported("fast activation requires CPU float16/float32 (strict precision: float16)")
        try:
            code = native_id(kind, parameters)
        except (ValueError, TypeError) as error:
            raise Unsupported(str(error)) from error
        if code is None and state.activation_policy == "native":
            raise Unsupported("native activation supports Softplus beta=1/threshold=20 and ELU alpha=1")
        _activation_capability(state.runtime, kind)
    except Unsupported as error:
        return state.fallback(path, str(error), original)
    native = (state.activation_policy == "native" or state.activation_policy == "auto"
              and auto_native(kind, input, parameters))
    native = native and code is not None
    if not native:
        output = original()
        state.report.record(path, "cpu_torch", "activation policy")
        return output
    # Elementwise GELU/SiLU keep input dtype under CPU autocast. Never apply
    # the projection autocast conversion or the legacy BF16 operand bridge.
    output = getattr(state.runtime, kind)(input, **parameters)
    state.report.record(path, "cpu_fast")
    return output


def _patch_activation(state, module, path, kind):
    original = module.forward
    def forward(this, input):
        return _activation(state, path, kind, input, **module_parameters(this, kind),
                           original=lambda: original(input), training=this.training)
    module.forward = types.MethodType(forward, module)
    state.undo.append(lambda: module.__dict__.pop("forward", None))


def optimize(model, *, runtime, strict=True, operators=("linear", "conv2d", "conv_transpose2d", "attention", *ACTIVATION_FUNCTIONS), predicate=None,
             precision="auto", activation_policy="auto"):
    """Adapt an eval Module in place, preserving parameters, aliases and state_dict.

    Strict coverage applies to recognized requested operators, not arbitrary
    Python/functional code. precision='auto' follows external FP16/FP32 dtypes,
    preserving parameter dtype and input/output precision. 'strict' requires
    FP16; legacy 'fp16' also permits the existing BF16-to-FP16 bridge.
    No global monkey-patches or device conversion.
    Call restore() before changing device, training, serialization or re-adapting.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("optimize expects nn.Module; pass a pipeline's transformer/unet separately")
    if model.training:
        raise ValueError("call model.eval() before optimize")
    if hasattr(model, "_tfllm_optimization"):
        raise ValueError("model already optimized; restore before re-optimizing")
    if precision not in ("auto", "strict", "fp16"):
        raise ValueError("precision must be auto, strict or fp16")
    operators = set(operators)
    if not operators or not operators <= {"linear", "conv2d", "conv_transpose2d", "attention", *ACTIVATION_FUNCTIONS}:
        raise ValueError("unknown operators; select projections, attention and/or " + ", ".join(ACTIVATION_FUNCTIONS))
    state = _State(runtime, strict, precision, activation_policy)
    planned = []
    # named_modules removes duplicate aliases: patch the shared instance once.
    modules = list(model.named_modules())
    try:
        from .integrations import plan_attention
        attention_plans, attention_ids = plan_attention(model, modules, state, operators, predicate)
        for path, module in modules:
            kind = ("linear" if isinstance(module, nn.Linear) else "conv2d" if isinstance(module, nn.Conv2d)
                    else "conv_transpose2d" if isinstance(module, nn.ConvTranspose2d)
                    else module_kind(module))
            if kind not in operators:
                continue
            if predicate is not None and not predicate(path, module):
                state.report.add(path, kind, "excluded", "predicate")
                continue
            if getattr(runtime,"attention_fast",False) and type(module) is nn.modules.linear.NonDynamicallyQuantizableLinear:
                state.report.add(path,kind,"retained","packed MHA functional projection")
                continue
            try:
                if "forward" in module.__dict__ or hasattr(module, "_hf_hook"):
                    raise Unsupported("custom/offload forward requires explicit adaptation")
                if module.training:
                    raise Unsupported("target module is in training mode")
                if kind in ACTIVATION_FUNCTIONS:
                    if type(module) not in ACTIVATION_MODULES:
                        raise Unsupported("custom activation subclasses require an explicit adapter")
                    _activation_capability(runtime, kind)
                    code = native_id(kind, module_parameters(module, kind))
                    if code is None and activation_policy == "native":
                        raise Unsupported("native activation supports Softplus beta=1/threshold=20 and ELU alpha=1")
                    planned.append((module, path, kind, ()))
                    state.report.add(path, kind, "routed", "CPU activation policy: " + activation_policy)
                    continue
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
                    fn=getattr(module.forward,"__func__",None)
                    if getattr(runtime,"attention_fast",False) and (type(module) is nn.MultiheadAttention or
                            (isinstance(fn,types.FunctionType) and "scaled_dot_product_attention" in fn.__code__.co_names)):
                        state.report.add(path,"attention","routed","FAST model-scoped functional SDPA")
                        continue
                    reason = "unrecognized attention module; select a supported integration or explicit graph backend"
                    state.report.add(path, "attention", "skipped", reason)
                    if strict:
                        raise Unsupported(f"{path or '<root>'}: {reason}")
        for module, path, kind, geometry in planned:
            if kind in ACTIVATION_FUNCTIONS:
                _patch_activation(state, module, path, kind)
            else:
                _patch_projection(state, module, path, kind, geometry)
        for install in attention_plans:
            install()
        if "attention" in operators and getattr(runtime,"attention_fast",False):
            from .integrations import _SDPAMode
            from .routing import AttentionRoute
            previous,had_forward=model.forward,"forward" in model.__dict__
            route=AttentionRoute(state,"<root>.sdpa")
            def forward(this,*args,**kwargs):
                with _SDPAMode(route):return previous(*args,**kwargs)
            model.forward=types.MethodType(forward,model)
            def undo_scope():
                if had_forward:model.forward=previous
                else:model.__dict__.pop("forward",None)
            state.undo.append(undo_scope)
            state.report.add("<root>.sdpa","attention","routed","FAST scoped SDPA; no process-global patch")
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
