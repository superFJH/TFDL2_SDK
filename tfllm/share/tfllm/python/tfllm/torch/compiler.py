"""Small Dynamo/FX backend: explicit lowering, no global dispatch patch."""
import math
import operator

import torch
from torch import nn
from torch.nn import functional as F

from .modules import _pair
from .optimize import _State, Unsupported, _activation
from .activation_ops import GRAPH_FUNCTIONS
from .routing import AttentionRoute


def _matmul(node):
    return isinstance(node, torch.fx.Node) and (
        node.op == 'call_function' and node.target in (operator.matmul, torch.matmul)
        or node.op == 'call_method' and node.target == 'matmul')


def _attention_pattern(node):
    """Only (Q @ K.transpose(-2,-1) * constant).softmax(-1) @ V.

    No masks, dropout, casts, GQA repeats or externally used score/probability
    tensors are guessed. SDPA is the preferred frontend for richer semantics.
    """
    if not _matmul(node) or len(node.args) != 2 or node.kwargs:
        return None
    probability, value = node.args
    if not isinstance(probability, torch.fx.Node) or len(probability.users) != 1:
        return None
    if not (probability.op == 'call_method' and probability.target == 'softmax'
            or probability.op == 'call_function' and probability.target in (F.softmax, torch.softmax)):
        return None
    if len(probability.args) > 2 or set(probability.kwargs) - {'dim', 'dtype', '_stacklevel'}:
        return None
    dim = probability.args[1] if len(probability.args) == 2 else probability.kwargs.get('dim')
    if dim != -1 or probability.kwargs.get('dtype') is not None:
        return None
    score, scale = probability.args[0], 1.0
    internals = [probability]
    if isinstance(score, torch.fx.Node) and score.op == 'call_function' and score.target in (operator.mul, operator.truediv, torch.mul, torch.div):
        if len(score.args) != 2 or score.kwargs or len(score.users) != 1:
            return None
        left, right = score.args
        if score.target in (operator.mul, torch.mul) and type(left) in (int, float):
            left, right = right, left
        if type(right) not in (int, float) or not math.isfinite(right) or right <= 0:
            return None
        scale = 1.0 / right if score.target in (operator.truediv, torch.div) else float(right)
        internals.append(score)
        score = left
    if not _matmul(score) or len(score.args) != 2 or score.kwargs or len(score.users) != 1:
        return None
    query, transposed = score.args
    if not isinstance(transposed, torch.fx.Node) or len(transposed.args) != 3 or transposed.kwargs:
        return None
    if not (transposed.op == 'call_method' and transposed.target == 'transpose'
            or transposed.op == 'call_function' and transposed.target == torch.transpose):
        return None
    if tuple(transposed.args[1:]) not in ((-2, -1), (-1, -2)):
        return None
    if len(transposed.users) != 1:
        return None
    return query, transposed.args[0], value, scale, internals + [score, transposed]


class CompilerBackend:
    def __init__(self, runtime, strict=True, activation_policy="auto"):
        self.state = _State(runtime, strict, activation_policy=activation_policy)
        self.report = self.state.report
        self.report.scope = 'captured FX matrix/SDPA/activation candidates only; custom ops and graph breaks are not audited'
        self.graphs = 0

    def _projection(self, path, kind):
        state = self.state
        def run(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            x = input
            def original():
                return F.linear(x, weight, bias) if kind == 'linear' else F.conv2d(x, weight, bias, stride, padding, dilation, groups)
            try:
                if torch.is_grad_enabled():
                    raise Unsupported('compiled NPU operators require no_grad()/inference_mode()')
                if x.dtype not in (torch.float16, torch.float32) or x.device.type != 'cpu':
                    raise Unsupported('activation must be CPU float16/float32')
                if bias is not None and (bias.dtype not in (torch.float16, torch.float32) or bias.device.type != 'cpu'):
                    raise Unsupported('bias must be CPU float16/float32')
                dtype = torch.get_autocast_dtype('cpu') if torch.is_autocast_enabled('cpu') else x.dtype
                if dtype not in (torch.float16, torch.float32):raise Unsupported('autocast output dtype is not supported')
                if kind == 'conv2d' and isinstance(padding, str):
                    raise Unsupported('string convolution padding is unsupported')
                geometry = () if kind == 'linear' else (_pair(stride), _pair(padding), _pair(dilation), groups)
                packed = state.weights.get(weight, kind, geometry)
            except (Unsupported, NotImplementedError) as error:
                return state.fallback(path, str(error), original)
            with torch.autocast('cpu', enabled=False):
                y = packed(x if x.dtype == dtype else x.to(dtype))
            state.report.record(path, 'npu' if state.runtime.backend == 'npu' else 'cpu_reference')
            if bias is None:return y
            bias = bias.to(y.dtype)
            return y + (bias if kind == 'linear' else bias[None, :, None, None])
        run.__name__ = 'tfllm_' + kind
        return run

    def _transpose_projection(self, path):
        state = self.state
        def run(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
            # This hardware boundary is explicit even in permissive mode.
            if _pair(stride) != (2, 2):raise Unsupported("ConvTranspose2d requires stride=2")
            def original():return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
            try:
                if torch.is_grad_enabled():raise Unsupported("compiled NPU operators require no_grad()/inference_mode()")
                if input.device.type != "cpu" or input.dtype not in (torch.float16, torch.float32):
                    raise Unsupported("activation must be CPU float16/float32")
                if bias is not None and (bias.device.type != "cpu" or bias.dtype not in (torch.float16, torch.float32)):
                    raise Unsupported("bias must be CPU float16/float32")
                dtype = torch.get_autocast_dtype('cpu') if torch.is_autocast_enabled('cpu') else input.dtype
                if dtype not in (torch.float16, torch.float32):raise Unsupported("autocast output dtype is not supported")
                geometry = (_pair(stride), _pair(padding), _pair(output_padding), groups, _pair(dilation))
                packed = state.weights.get(weight, 'conv_transpose2d', geometry)
            except (Unsupported, NotImplementedError) as error:
                return state.fallback(path, str(error), original)
            with torch.autocast('cpu', enabled=False):
                y = packed(input if input.dtype == dtype else input.to(dtype))
            state.report.record(path, 'npu' if state.runtime.backend == 'npu' else 'cpu_reference')
            return y if bias is None else y+bias.to(y.dtype)[None, :, None, None]
        run.__name__ = 'tfllm_conv_transpose2d'
        return run

    def __call__(self, gm, example_inputs):
        # Dynamo graph inputs remain live: never freeze its example tensors or
        # physical addresses. Weight identity/version is checked on execution.
        prefix = f'graph{self.graphs}'
        self.graphs += 1
        graph = gm.graph
        fused = set()
        for node in list(graph.nodes):
            match = _attention_pattern(node)
            if match is None:
                continue
            q, k, v, scale, internals = match
            path = prefix + '.' + node.name
            route = AttentionRoute(self.state, path)
            def attention(q, k, v, scale=scale, route=route):
                return route(q, k, v, scale=scale)
            attention.__name__ = 'tfllm_fused_attention'
            node.op, node.target = 'call_function', attention
            node.args, node.kwargs = (q, k, v), {}
            for old in internals:
                graph.erase_node(old)
            fused.add(node)
            self.report.add(path, 'attention', 'routed', 'complete QK/scale/softmax/AV pattern')
        for node in list(graph.nodes):
            if node in fused:
                continue
            path = prefix + '.' + node.name
            kind = None
            activation_spec = GRAPH_FUNCTIONS.get(node.target) if node.op == 'call_function' else None
            if node.op == 'call_method' and node.target in ('sigmoid', 'tanh'):
                activation_spec = (node.target, ())
            if activation_spec:
                activation_kind, positional = activation_spec
                def activate(input, *args, path=path, kind=activation_kind, positional=positional, **kwargs):
                    if len(args) > len(positional):raise TypeError('unexpected activation positional arguments')
                    for key, value in zip(positional, args):
                        if key in kwargs:raise TypeError('duplicate activation argument: ' + key)
                        kwargs[key] = value
                    return _activation(self.state, path, kind, input, **kwargs)
                activate.__name__ = 'tfllm_' + activation_kind
                node.op, node.target = 'call_function', activate
                self.report.add(path, activation_kind, 'routed', 'CPU activation policy')
                continue
            if node.op == 'call_function':
                if node.target is F.linear:
                    kind = 'linear'
                elif node.target in (F.conv2d, torch.conv2d):
                    kind = 'conv2d'
                elif node.target in (F.conv_transpose2d, torch.conv_transpose2d):
                    stride = node.args[3] if len(node.args)>3 else node.kwargs.get('stride', 1)
                    if not isinstance(stride, (int, tuple, list)) or _pair(stride) != (2, 2):
                        raise Unsupported(path + ': ConvTranspose2d requires stride=2')
                    node.target = self._transpose_projection(path)
                    self.report.add(path, 'conv_transpose2d', 'routed', 'native stride-2 deconvolution')
                    continue
                elif node.target is F.scaled_dot_product_attention:
                    route = AttentionRoute(self.state, path)
                    def sdpa(*args, route=route, **kwargs):
                        return route(*args, **kwargs)
                    sdpa.__name__ = 'tfllm_sdpa'
                    node.target = sdpa
                    self.report.add(path, 'attention', 'routed', 'SDPA')
                    continue
            if kind:
                node.target = self._projection(path, kind)
                self.report.add(path, kind, 'routed', 'functional operator')
            elif _matmul(node) or (node.op == 'call_function' and node.target in (torch.bmm, torch.mm)) or (
                    node.op == 'call_method' and node.target in ('bmm', 'mm')) or node.op == 'call_module' and isinstance(gm.get_submodule(node.target), (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.MultiheadAttention)):
                reason = 'unmatched matrix/attention operation; use SDPA or an explicit adapter'
                self.report.add(path, 'matrix', 'skipped', reason)
                if self.state.strict:
                    raise Unsupported(path + ': ' + reason)
        graph.lint()
        gm.recompile()
        return gm.forward

    def close(self):
        """Drop packed snapshots; do not call concurrently with inference."""
        self.state.weights.clear()


def make_backend(*, runtime, strict=True, activation_policy="auto"):
    """Use torch.compile(model, backend=make_backend(...), fullgraph=True).

    This backend executes untouched operators with PyTorch. Strict applies to
    recognized matrix/attention candidates inside captured graphs, not arbitrary
    custom ops or graph breaks. Do not combine with optimize() on the same model.
    """
    return CompilerBackend(runtime, strict, activation_policy)
