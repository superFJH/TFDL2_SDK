from ..TFConvertor import TFConvertor
import numpy as np
from .ActivationPatterns import graph_edges, other_operand, scalar_is, private_intermediates


def MergeGeLUOnnx(Model:TFConvertor):
    """Fuse the complete erf GeLU formula without crossing Q/DQ or fan-out."""
    producers, consumers = graph_edges(Model)
    count = 0
    for erf in list(Model.nodes.values()):
        if erf.op != "Erf" or len(erf.inputs) != 1:
            continue
        scaled = producers.get(erf.inputs[0])
        if scaled is None or len(scaled.inputs) != 2:
            continue
        if scaled.op == "Div" and scalar_is(Model, scaled.inputs[1], np.sqrt(2.)):
            x = scaled.inputs[0]
        elif scaled.op == "Mul":
            constants = [n for n in scaled.inputs if scalar_is(Model, n, 1/np.sqrt(2.))]
            if len(constants) != 1:
                continue
            x = other_operand(scaled, constants[0])
        else:
            continue
        users = consumers.get(erf.outputs[0], [])
        if len(users) != 1 or users[0].op != "Add":
            continue
        add = users[0]
        if not scalar_is(Model, other_operand(add, erf.outputs[0]), 1.):
            continue
        users = consumers.get(add.outputs[0], [])
        if len(users) != 1 or users[0].op != "Mul":
            continue
        first = users[0]
        users = consumers.get(first.outputs[0], [])
        if len(users) != 1 or users[0].op != "Mul":
            continue
        final = users[0]
        a = other_operand(first, add.outputs[0])
        b = other_operand(final, first.outputs[0])
        if not ((a == x and scalar_is(Model, b, .5)) or
                (b == x and scalar_is(Model, a, .5))):
            continue
        matched = [scaled, erf, add, first, final]
        if not private_intermediates(Model, consumers, matched, final):
            continue
        final.inputs, final.op, final.attr = [x], "GeLU", {}
        for node in matched[:-1]:
            Model.nodes.pop(node.name)
        count += 1
    # Constants may be shared by Q/DQ or another activation; do not delete them.
    return count
