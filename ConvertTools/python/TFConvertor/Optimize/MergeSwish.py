from ..TFConvertor import TFConvertor
from .ActivationPatterns import graph_edges, other_operand, private_intermediates


def MergeSwishOnnx(Model:TFConvertor):
    _, consumers = graph_edges(Model)
    count = 0
    for sigmoid in list(Model.nodes.values()):
        if sigmoid.op != "Sigmoid":
            continue
        users = consumers.get(sigmoid.outputs[0], [])
        if len(users) != 1 or users[0].op != "Mul":
            continue
        mul = users[0]
        if other_operand(mul, sigmoid.outputs[0]) != sigmoid.inputs[0]:
            continue
        if not private_intermediates(Model, consumers, [sigmoid, mul], mul):
            continue
        mul.inputs, mul.op, mul.attr = sigmoid.inputs[:], "Swish", {}
        Model.nodes.pop(sigmoid.name)
        count += 1
    return count
