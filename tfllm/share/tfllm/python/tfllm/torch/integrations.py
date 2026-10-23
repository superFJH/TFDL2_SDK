"""Allowlisted framework adapters; optional dependencies are imported lazily."""
import copy
import importlib
import inspect
import threading
import types as _function_types

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode

from .optimize import Unsupported
from .routing import AttentionRoute, DeferredMask, IntervalMask, materialize_mask, attention_fallback

_registration_lock = threading.Lock()


def _hf_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0,
             mask_function=None, attention_mask=None, **kwargs):
    from transformers import masking_utils as masks
    def deferred(reason):
        return DeferredMask(reason, lambda: masks.sdpa_mask(batch_size, q_length, kv_length,
            q_offset=q_offset, kv_offset=kv_offset, mask_function=mask_function,
            attention_mask=attention_mask, **kwargs))
    # Never guess the meaning of composed/packed/sliding/custom mask callbacks.
    if mask_function not in (masks.causal_mask_function, masks.bidirectional_mask_function):
        return deferred("custom/packed/sliding Transformers mask is not yet supported")
    if str(kwargs.get("device", "cpu")) != "cpu":
        return deferred("Transformers mask must be on CPU")
    qo, ko = int(q_offset), int(kv_offset)
    if min(qo, ko) < 0:
        return deferred("negative mask offsets are unsupported")
    bounds = torch.empty((batch_size, q_length, 2), dtype=torch.int64)
    bounds[..., 0] = 0
    bounds[..., 1] = (torch.arange(q_length) + qo - ko + 1).clamp(0, kv_length) if mask_function is masks.causal_mask_function else kv_length
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size or attention_mask.device.type != "cpu":
            return deferred("unsupported Transformers padding mask layout")
        for b in range(batch_size):
            row = attention_mask[b, ko:ko + kv_length]
            if row.numel() != kv_length:
                return deferred("padding mask does not cover physical KV length")
            if not bool(((row == 0) | (row == 1)).all()):
                return deferred("padding mask must contain only 0/1")
            ids = row.nonzero().flatten()
            first, end = (int(ids[0]), int(ids[-1]) + 1) if len(ids) else (0, 0)
            if len(ids) != end - first:
                return deferred("padding holes require non-contiguous attention")
            bounds[b, :, 1].clamp_(max=end)
            bounds[b, :, 0] = torch.minimum(torch.full((q_length,), first), bounds[b, :, 1])
    return IntervalMask(bounds)


def _hf_attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
    state, path, fallback = module._tfllm_attention_route
    def original():
        mask = materialize_mask(attention_mask, key.shape[-2])
        if fallback.__name__ == "eager_attention_forward" and mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill(~mask, -torch.inf)
        return fallback(module, query, key, value, mask, dropout=dropout, scaling=scaling, **kwargs)
    forbidden = ("output_attentions", "position_bias", "softcap", "soft_cap", "sliding_window")
    if module.training or any(kwargs.get(k) is not None and kwargs[k] is not False for k in forbidden):
        return attention_fallback(state,path,"training/attention weights/bias/softcap/window require original attention",original)
    if isinstance(attention_mask, DeferredMask):
        if getattr(state.runtime,"attention_fast",False):attention_mask=materialize_mask(attention_mask,key.shape[-2])
        else:return state.fallback(path, attention_mask.reason, original)
    # The registered mask builder preserves absolute query/KV offsets. None is
    # only interpreted like HF SDPA when callers intentionally bypass it.
    causal = bool(kwargs.get("is_causal", getattr(module, "is_causal", True))) and query.shape[2] > 1 and attention_mask is None
    route = AttentionRoute(state, path)
    out = route(query, key, value, attention_mask, dropout, causal, scale=scaling,
                enable_gqa=query.shape[1] != key.shape[1])
    return out.transpose(1, 2).contiguous(), None


class _SDPAMode(TorchFunctionMode):
    def __init__(self, route, *, unused_mha_weights=False):
        super().__init__()
        self.route, self.calls = route, 0
        self.unused_mha_weights = unused_mha_weights

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is F.scaled_dot_product_attention:
            self.calls += 1
            return self.route(*args, **(kwargs or {}))
        # Packed MHA helpers dispatch the entire helper through TorchFunctionMode,
        # otherwise hiding their inner SDPA while the mode is temporarily popped.
        packed = (func is F.multi_head_attention_forward or
                  (getattr(func,"__module__",None)=="sam3.model.model_misc" and
                   getattr(func,"__name__",None)=="multi_head_attention_forward"))
        if packed and getattr(self.route.state.runtime,"attention_fast",False):
            bound=inspect.signature(func).bind_partial(*args,**(kwargs or {}))
            need=bound.arguments.get("need_weights",inspect.signature(func).parameters["need_weights"].default)
            if need and not self.unused_mha_weights:
                raise Unsupported("FAST attention does not materialize attention weights; pass need_weights=False")
            if need:
                bound.arguments['need_weights']=False
                args,kwargs=bound.args,bound.kwargs
            local=_function_types.FunctionType(func.__code__,dict(func.__globals__,has_torch_function=lambda values:False),
                func.__name__,func.__defaults__,func.__closure__)
            local.__kwdefaults__=func.__kwdefaults__
            with _SDPAMode(self.route,unused_mha_weights=self.unused_mha_weights):return local(*args, **(kwargs or {}))
        return func(*args, **(kwargs or {}))


class _DiffusersProcessor:
    def __init__(self, original, route):
        self.original, self.route = original, route

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        if attn.training:
            return attention_fallback(self.route.state,self.route.path,"attention is in training mode",self.original,
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
        # Keep upstream normalization, layout, residual, projections and masks.
        # Only this invocation's SDPA is intercepted; no process-global patch.
        with _SDPAMode(self.route) as mode:
            out = self.original(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
        if not mode.calls:
            raise Unsupported("Diffusers processor did not call SDPA; adapter/version mismatch")
        return out


def install_fast_sam3_decoder(state,path,module):
    """This audited SAM3 layer consumes only MHA output [0], never weights."""
    if "forward" in module.__dict__:
        raise Unsupported(f"{path}: custom SAM3 decoder forward requires an explicit adapter")
    original=module.forward
    route=AttentionRoute(state,path)
    def forward(this,*args,**kwargs):
        with _SDPAMode(route,unused_mha_weights=True):return original(*args,**kwargs)
    module.forward=_function_types.MethodType(forward,module)
    state.undo.append(lambda:module.__dict__.pop("forward",None))
    state.report.add(path,"attention","routed","SAM3 decoder discards MHA weights; use SDPA output")


def install_fast_sam3_mha(state,path,module,allow_override=False):
    """Keep SAM3 helper kwargs before its TorchFunction dispatch drops bias.

    Function copies keep the provider's mask/layout code and its original
    globals; only the outer override-dispatch check is disabled locally.
    """
    if not allow_override and "forward" in module.__dict__:
        raise Unsupported(f"{path}: custom SAM3 MHA forward requires an explicit adapter")
    original,had=module.forward,"forward" in module.__dict__
    vendor=type(module).forward;helper=vendor.__globals__["multi_head_attention_forward"]
    local_helper=_function_types.FunctionType(helper.__code__,dict(helper.__globals__,has_torch_function=lambda values:False),
        helper.__name__,helper.__defaults__,helper.__closure__)
    local_helper.__kwdefaults__=helper.__kwdefaults__
    local_forward=_function_types.FunctionType(vendor.__code__,dict(vendor.__globals__,multi_head_attention_forward=local_helper),
        vendor.__name__,vendor.__defaults__,vendor.__closure__)
    local_forward.__kwdefaults__=vendor.__kwdefaults__
    route=AttentionRoute(state,path)
    def forward(this,*args,**kwargs):
        bound=inspect.signature(vendor).bind_partial(this,*args,**kwargs)
        if bound.arguments.get("need_weights",False):
            raise Unsupported("FAST attention does not materialize attention weights; pass need_weights=False")
        with _SDPAMode(route):return local_forward(this,*args,**kwargs)
    module.forward=_function_types.MethodType(forward,module)
    def undo():
        if had:module.forward=original
        else:module.__dict__.pop("forward",None)
    state.undo.append(undo)
    state.report.add(path,"attention","routed","FAST packed MHA; original mask/bias/layout preserved")


def plan_attention(model, modules, state, operators, predicate):
    if "attention" not in operators:
        return [], set()
    plans, handled = [], set()
    hf = []
    families = {"llama": "LlamaAttention", "qwen2": "Qwen2Attention", "qwen3": "Qwen3Attention"}
    for path, module in modules:
        cls = type(module)
        if getattr(state.runtime,"attention_fast",False) and (cls.__module__,cls.__name__)==("sam3.model.decoder","TransformerDecoderLayer"):
            if predicate is None or predicate(path,module):
                handled.add(id(module))
                plans.append(lambda path=path,module=module:install_fast_sam3_decoder(state,path,module))
            continue
        if getattr(state.runtime,"attention_fast",False) and (cls.__module__,cls.__name__)==("sam3.model.model_misc","MultiheadAttention"):
            if predicate is None or predicate(path,module):
                handled.add(id(module))
                plans.append(lambda path=path,module=module:install_fast_sam3_mha(state,path,module))
            continue
        if cls.__module__.startswith("transformers.models.") and "attention" in cls.__name__.lower():
            hf.append((path, module))
        if cls.__module__ == "diffusers.models.attention_processor" and cls.__name__ == "Attention":
            if predicate is not None and not predicate(path, module):
                continue
            handled.add(id(module))
            from diffusers.models.attention_processor import AttnProcessor2_0
            if type(module.processor) is not AttnProcessor2_0 or "forward" in module.__dict__ or hasattr(module, "_hf_hook"):
                reason = "only unmodified Diffusers Attention + AttnProcessor2_0 is supported"
                state.report.add(path, "attention", "skipped", reason)
                if state.strict:
                    raise Unsupported(f"{path}: {reason}")
                continue
            def install(module=module, path=path):
                old = module.processor
                module.set_processor(_DiffusersProcessor(old, AttentionRoute(state, path)))
                state.undo.append(lambda: module.set_processor(old))
            plans.append(install)
            state.report.add(path, "attention", "routed", "Diffusers SDPA processor")
    if hf:
        from transformers import AttentionInterface, AttentionMaskInterface, masking_utils
        selected = [item for item in hf if predicate is None or predicate(*item)]
        reason = None
        if len(selected) != len(hf):
            reason = "Transformers attention selection must cover the whole model (shared mask configuration)"
        for path, module in hf:
            family = getattr(module.config, "model_type", None)
            if (family not in families or type(module).__name__ != families[family]
                    or type(module).__module__ != f"transformers.models.{family}.modeling_{family}"
                    or "forward" in module.__dict__ or hasattr(module, "_hf_hook")):
                reason = "supported Transformers attention families: unmodified Llama/Qwen2/Qwen3"
        if "q_length" not in inspect.signature(masking_utils.sdpa_mask).parameters:
            reason = "unsupported Transformers mask API; tested with 5.17"
        if reason:
            for path, module in hf:
                handled.add(id(module))
                state.report.add(path, "attention", "skipped", reason)
            if state.strict and selected:
                raise Unsupported(reason)
        else:
            # Register stateless callbacks once. Runtime selection is scoped to
            # each model, not captured in a global registry closure.
            with _registration_lock:
                AttentionInterface.register("tfllm", _hf_attention)
                AttentionMaskInterface.register("tfllm", _hf_mask)
            fallbacks = {}
            for path, module in hf:
                previous = module.config._attn_implementation or "eager"
                if previous == "eager":
                    fallback = getattr(importlib.import_module(type(module).__module__), "eager_attention_forward")
                else:
                    fallback = AttentionInterface()[previous]
                fallbacks[id(module)] = fallback
                handled.add(id(module))
                state.report.add(path, "attention", "routed", "Transformers compact causal/padding mask")
            def install_hf():
                config_ids = {id(m.config) for _, m in hf}
                copies = {}
                for _, module in modules:
                    old = getattr(module, "config", None)
                    if id(old) in config_ids:
                        if id(old) not in copies:
                            copies[id(old)] = copy.copy(old)
                            copies[id(old)]._attn_implementation = "tfllm"
                        module.config = copies[id(old)]
                        state.undo.append(lambda module=module, old=old: setattr(module, "config", old))
                for path, module in hf:
                    module._tfllm_attention_route = (state, path, fallbacks[id(module)])
                    state.undo.append(lambda module=module: module.__dict__.pop("_tfllm_attention_route", None))
            plans.append(install_hf)
    return plans, handled
