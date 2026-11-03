"""Fuse the exact GAP -> two FC -> sigmoid -> multiply SE motif.

This pass runs after Gemm weights have been normalized to [out, in]. It checks
data edges, single-use intermediates, inferred shapes, and FP32 parameters;
node names are not used to recognize a block.
"""

from collections import Counter, OrderedDict

import numpy as np
from TFDL2 import Op

from ..onnxConvertor import OnnxNodeConvertor


def convert_se_block(*args, node):
    if len(args) != 5:
        raise ValueError("SEBlock expects activation, FC1 weight/bias, FC2 weight/bias")
    result = Op.Custom(args, (node.outputs[0],), "SEBlock", '{"outputType":"auto"}')
    return result[0] if isinstance(result, (tuple, list)) else result


def merge_se_blocks(model):
    """Replace safe, structurally matched SE branches with one Custom op."""
    nodes = list(model.nodes.values())
    producer = {output: node for node in nodes for output in node.outputs}
    consumers = Counter(name for node in nodes for name in node.inputs)
    graph_outputs = set(model.outputs)
    shapes = getattr(model, "tensor_shapes", {})
    matches = []
    occupied = set()

    def parent(name, operation):
        node = producer.get(name)
        return node if node is not None and node.op == operation else None

    for mul in nodes:
        if mul.op != "Mul" or len(mul.inputs) != 2 or not mul.outputs:
            continue
        for gate_index in (0, 1):
            expand = parent(mul.inputs[gate_index], "Expand")
            if expand is None or len(expand.inputs) < 2:
                continue
            reshape = parent(expand.inputs[0], "Reshape")
            sigmoid = parent(reshape.inputs[0], "Sigmoid") if reshape and reshape.inputs else None
            second = parent(sigmoid.inputs[0], "Gemm") if sigmoid and sigmoid.inputs else None
            relu = parent(second.inputs[0], "Relu") if second and second.inputs else None
            first = parent(relu.inputs[0], "Gemm") if relu and relu.inputs else None
            flatten = parent(first.inputs[0], "Flatten") if first and first.inputs else None
            pool = parent(flatten.inputs[0], "GlobalAveragePool") if flatten and flatten.inputs else None
            chain = [pool, flatten, first, relu, second, sigmoid, reshape, expand]
            if any(node is None or len(node.outputs) != 1 for node in chain):
                continue
            activation = pool.inputs[0] if len(pool.inputs) == 1 else None
            if activation is None or mul.inputs[1 - gate_index] != activation:
                continue
            if any(consumers[node.outputs[0]] != 1 or node.outputs[0] in graph_outputs
                   for node in chain):
                continue
            if first.inputs[0] != flatten.outputs[0] or second.inputs[0] != relu.outputs[0]:
                continue
            if len(first.inputs) != 3 or len(second.inputs) != 3:
                continue
            if flatten.attr.get("axis", 1) != 1:
                continue
            if first.attr.get("transA", 0) != 0 or second.attr.get("transA", 0) != 0:
                continue
            keys = [first.inputs[1], first.inputs[2], second.inputs[1], second.inputs[2]]
            if any(key not in model.params for key in keys):
                continue
            weights = [np.asarray(model.params[key]) for key in keys]
            if any(value.dtype != np.float32 for value in weights):
                continue
            w1, b1, w2, b2 = weights
            x_shape = shapes.get(activation)
            gate_shape = shapes.get(reshape.outputs[0])
            expand_shape = shapes.get(expand.outputs[0])
            output_shape = shapes.get(mul.outputs[0])
            if (x_shape is None or len(x_shape) != 4
                    or any(not isinstance(dim, (int, np.integer)) or dim < 1
                           for dim in x_shape)
                    or gate_shape != [x_shape[0], x_shape[1], 1, 1]
                    or expand_shape != x_shape or output_shape != x_shape):
                continue
            channels = x_shape[1]
            if (w1.ndim != 2 or w1.shape[1] != channels or w1.shape[0] < 1
                    or b1.shape != (w1.shape[0],)
                    or w2.shape != (channels, w1.shape[0])
                    or b2.shape != (channels,)):
                continue
            names = {node.name for node in chain + [mul]}
            if names & occupied:
                continue
            replacement = OnnxNodeConvertor(graph=model)
            replacement.name = mul.name + "__SEBlock"
            replacement.op = "SEBlock"
            replacement.inputs = [activation, *keys]
            replacement.outputs = list(mul.outputs)
            replacement.attr = {}
            matches.append((pool.name, names, replacement))
            occupied.update(names)
            break

    replacements = {first: replacement for first, _, replacement in matches}
    removed = set().union(*(names for _, names, _ in matches)) if matches else set()
    rebuilt = OrderedDict()
    for node in nodes:
        if node.name in replacements:
            replacement = replacements[node.name]
            rebuilt[replacement.name] = replacement
        if node.name not in removed:
            rebuilt[node.name] = node
    model.nodes = rebuilt
    model.se_block_fusion = {
        "count": len(matches),
        "replaced_node_count": len(removed),
        "custom_nodes": [replacement.name for _, _, replacement in matches],
    }
    return len(matches)
