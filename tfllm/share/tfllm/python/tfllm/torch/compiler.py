"""Small Dynamo/FX backend: explicit lowering, no global dispatch patch."""
import math
import operator

import torch
from torch import nn
from torch.nn import functional as F

from .modules import _pair
from .optimize import _State, Unsupported
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
    def __init__(self, runtime, strict=True):
        self.state = _State(runtime, strict)
        self.report = self.state.report
        self.report.scope = 'captured FX matrix/SDPA candidates only; custom ops and graph breaks are not audited'
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
                if x.dtype != torch.float16 or x.device.type != 'cpu':
                    raise Unsupported('activation must be CPU float16')
                if bias is not None and (bias.dtype != torch.float16 or bias.device.type != 'cpu'):
                    raise Unsupported('bias must be CPU float16')
                if kind == 'conv2d' and isinstance(padding, str):
                    raise Unsupported('string convolution padding is unsupported')
                geometry = () if kind == 'linear' else (_pair(stride), _pair(padding), _pair(dilation), groups)
                packed = state.weights.get(weight, kind, geometry)
            except (Unsupported, NotImplementedError) as error:
                return state.fallback(path, str(error), original)
            y = packed(x)
            state.report.record(path, 'npu' if state.runtime.backend == 'npu' else 'cpu_reference')
            return y if bias is None else y + (bias if kind == 'linear' else bias[None, :, None, None])
        run.__name__ = 'tfllm_' + kind
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
            if node.op == 'call_function':
                if node.target is F.linear:
                    kind = 'linear'
                elif node.target in (F.conv2d, torch.conv2d):
                    kind = 'conv2d'
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
                    node.op == 'call_method' and node.target in ('bmm', 'mm')) or node.op == 'call_module' and isinstance(gm.get_submodule(node.target), (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
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


def make_backend(*, runtime, strict=True):
    """Use torch.compile(model, backend=make_backend(...), fullgraph=True).

    This backend executes untouched operators with PyTorch. Strict applies to
    recognized matrix/attention candidates inside captured graphs, not arbitrary
    custom ops or graph breaks. Do not combine with optimize() on the same model.
    """
    return CompilerBackend(runtime, strict)
