"""Audited manual attention adapters; retain positions, prompts and masks."""
import contextvars
import types

import torch

from ..torch.routing import AttentionRoute, attention_fallback


_SAM_SDPA = {
    'sam2.modeling.backbones.hieradet': {'MultiScaleAttention'},
    'sam2.modeling.sam.transformer': {'Attention','RoPEAttention'},
    'sam3.model.vitdet': {'Attention'},
    'sam3.model.model_misc': {'MultiheadAttention'},
    'sam3.model.decoder': {'SimpleRoPEAttention'},
}
_SAM_ROOTS = {('sam2.modeling.sam2_base','SAM2Base'),('sam3.model.sam3_image','Sam3Image')}


def _sam_cpu_attention(module,args,kwargs,runtime):
    """Skip interception when every possible SDPA in this module is CPU."""
    if getattr(runtime,"attention_fast",False):return False
    q=args[0] if args else kwargs.get('q',kwargs.get('query',kwargs.get('x')))
    if not isinstance(q,torch.Tensor) or q.device.type!='cpu' or module.training:return False
    dtype=torch.get_autocast_dtype('cpu') if torch.is_autocast_enabled('cpu') else q.dtype
    if dtype!=torch.float32:return False
    if runtime.attention_policy=='cpu':return True
    if runtime.attention_policy!='hybrid':return False
    kind=type(module).__module__
    if kind=='sam3.model.vitdet':
        length=q.shape[1]*q.shape[2] if q.ndim==4 else q.shape[-2]
        return length<2048
    k=args[1] if len(args)>1 else kwargs.get('k',kwargs.get('key',q))
    if not isinstance(k,torch.Tensor) or q.ndim not in (2,3) or k.ndim!=q.ndim:return False
    heads=getattr(module,'num_heads',0)
    if not heads:return False
    if kind=='sam3.model.model_misc':
        axis=1 if q.ndim==3 and module.batch_first else 0
        dim=module.head_dim
    else:
        axis=-2;dim=getattr(module,'internal_dim',q.shape[-1])//heads
    return not (48<=dim<=256 and q.shape[axis]>=1024 and k.shape[axis]>=1024
                and q.shape[axis]*k.shape[axis]>=4*1024*1024)


def _install_sam_scopes(model,state):
    """Keep SDPA interception inside audited official attention modules.

    Raw processor/model calls stay original, as before VisionSession routing.
    Hiera's simple projection/pooling formula has a direct adapter; the other
    classes keep their original forward, including RoPE and relative positions.
    """
    if (type(model).__module__,type(model).__name__) not in _SAM_ROOTS or 'forward' in model.__dict__:
        return set()
    from ..torch.integrations import _SDPAMode
    active=contextvars.ContextVar('tfllm_vision_attention_active',default=False)
    handled=set();unsafe=False;backbone=False
    undo_start=len(state.undo);report_start=len(state.report.entries)
    for path,module in model.named_modules():
        cls=type(module);function=getattr(module.forward,'__func__',None)
        recognized=cls.__name__ in _SAM_SDPA.get(cls.__module__,set())
        if not recognized:
            if isinstance(function,types.FunctionType) and 'scaled_dot_product_attention' in function.__code__.co_names:
                unsafe=True
            continue
        if 'forward' in module.__dict__ or not isinstance(function,types.FunctionType):
            unsafe=True;continue
        original=module.forward;route=AttentionRoute(state,path)
        if cls.__module__=='sam2.modeling.backbones.hieradet' and all(
                hasattr(module,name) for name in ('qkv','proj','q_pool','num_heads')):
            pool=function.__globals__.get('do_pool')
            if module.q_pool is not None and pool is None:
                unsafe=True;continue
            def forward(this,x,_original=original,_route=route,_pool=pool):
                if not active.get():return _original(x)
                if this.training:return attention_fallback(state,_route.path,'training Hiera attention remains original',_original,x)
                b,h,w,_=x.shape
                q,k,v=torch.unbind(this.qkv(x).reshape(b,h*w,3,this.num_heads,-1),2)
                if this.q_pool:
                    q=_pool(q.reshape(b,h,w,-1),this.q_pool);h,w=q.shape[1:3]
                    q=q.reshape(b,h*w,this.num_heads,-1)
                y=_route(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
                return this.proj(y.transpose(1,2).reshape(b,h,w,-1))
        else:
            def forward(this,*args,_original=original,_route=route,**kwargs):
                if not active.get():return _original(*args,**kwargs)
                if state.precision!='strict' and _sam_cpu_attention(this,args,kwargs,state.runtime):
                    state.report.record(_route.path,'cpu_attention','original SAM module selected by attention policy')
                    return _original(*args,**kwargs)
                with _SDPAMode(_route):return _original(*args,**kwargs)
        module.forward=types.MethodType(forward,module)
        state.undo.append(lambda module=module:module.__dict__.pop('forward',None))
        state.report.add(path,'attention','routed','official SAM module scope; no process-global patch')
        handled.add(id(module))
        backbone |= cls.__module__ in ('sam2.modeling.backbones.hieradet','sam3.model.vitdet')
    if unsafe or not backbone:
        for undo in reversed(state.undo[undo_start:]):undo()
        del state.undo[undo_start:];del state.report.entries[report_start:]
        return set()
    if handled:
        state.vision_attention_context=active
        state.vision_module_attention=True
    return handled


def install(model, state):
    handled = _install_sam_scopes(model,state)
    for path, module in model.named_modules():
        cls = type(module)
        dino = cls.__module__ in {
            "depth_anything_v2.dinov2_layers.attention", "dinov2.layers.attention",
            "depth_anything.dinov2_layers.attention",
        } and cls.__name__ in {"Attention", "MemEffAttention"}
        sam = cls.__module__ == "segment_anything.modeling.transformer" and cls.__name__ == "Attention"
        if not (dino or sam):
            continue
        original = module.forward
        if "forward" in module.__dict__:
            state.report.add(path, "attention", "skipped", "custom instance forward")
            continue
        route = AttentionRoute(state, path)
        if dino:
            required = ("qkv", "proj", "proj_drop", "attn_drop", "num_heads", "scale")
            if not all(hasattr(module, a) for a in required):
                raise ValueError(f"{path}: unsupported DINOv2 attention API")
            if any(hasattr(module, a) for a in ("q_norm", "k_norm", "rope", "relative_position_bias")):
                state.report.add(path, "attention", "skipped", "DINOv2 normalization/position extension requires original attention")
                continue

            def forward(this, x, attn_bias=None, _route=route, _original=original):
                if this.training or attn_bias is not None and not (getattr(state.runtime,'attention_fast',False) and isinstance(attn_bias,torch.Tensor)):
                    args = (x,) if attn_bias is None else (x, attn_bias)
                    return attention_fallback(state,_route.path,"training/nested DINOv2 attention remains original",_original,*args)
                b, n, c = x.shape
                q, k, v = this.qkv(x).reshape(b, n, 3, this.num_heads, c // this.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
                y = _route(q, k, v, attn_mask=attn_bias, scale=this.scale).transpose(1, 2).reshape(b, n, c)
                return this.proj_drop(this.proj(y))
        else:
            required = ("q_proj", "k_proj", "v_proj", "out_proj", "num_heads", "_separate_heads", "_recombine_heads")
            if not all(hasattr(module, a) for a in required):
                raise ValueError(f"{path}: unsupported SAM decoder attention API")

            def forward(this, q, k, v, _route=route, _original=original):
                if this.training:
                    return attention_fallback(state,_route.path,"training SAM attention remains original",_original,q,k,v)
                q = this._separate_heads(this.q_proj(q), this.num_heads)
                k = this._separate_heads(this.k_proj(k), this.num_heads)
                v = this._separate_heads(this.v_proj(v), this.num_heads)
                return this.out_proj(this._recombine_heads(_route(q, k, v)))
        module.forward = types.MethodType(forward, module)
        handled.add(id(module))
        # Restoring a class-defined forward must not leave a stale instance override.
        state.undo.append(lambda module=module: module.__dict__.pop("forward", None))
        state.report.add(path, "attention", "routed", "audited DINOv2/SAM decoder SDPA; original projections retained")
    if getattr(state.runtime,"attention_fast",False):
        from ..torch.integrations import install_fast_sam3_mha, install_fast_sam3_decoder
        for path,module in model.named_modules():
            if (type(module).__module__,type(module).__name__)==("sam3.model.decoder","TransformerDecoderLayer"):
                install_fast_sam3_decoder(state,path,module)
                handled.add(id(module))
            if (type(module).__module__,type(module).__name__)==("sam3.model.model_misc","MultiheadAttention"):
                install_fast_sam3_mha(state,path,module,allow_override=id(module) in handled)
                handled.add(id(module))
    return handled
