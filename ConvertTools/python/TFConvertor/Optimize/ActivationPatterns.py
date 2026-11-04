"""Helpers for local activation fusions that preserve shared graph edges."""
import numpy as np


def graph_edges(model):
    producers, consumers = {}, {}
    for node in model.nodes.values():
        for output in node.outputs:
            producers[output] = node
        for name in node.inputs:
            consumers.setdefault(name, []).append(node)
    return producers, consumers


def other_operand(node, name):
    if len(node.inputs) != 2 or name not in node.inputs:
        return None
    return node.inputs[1] if node.inputs[0] == name else node.inputs[0]


def scalar_is(model, name, value):
    if name not in model.params:
        return False
    actual = np.asarray(model.params[name])
    if actual.size != 1:
        return False
    # Accept the scalar represented in FP32 or FP64, not a different formula.
    actual = float(actual.reshape(-1)[0])
    return actual in (float(value), float(np.float32(value)))


def private_intermediates(model, consumers, nodes, final):
    """Every removed output must be private to this complete matched subgraph."""
    names = {node.name for node in nodes}
    public = set(model.outputs)
    return all(output not in public and
               all(user.name in names for user in consumers.get(output, []))
               for node in nodes if node is not final for output in node.outputs)
