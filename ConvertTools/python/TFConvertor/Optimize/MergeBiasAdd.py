from ..TFConvertor import TFConvertor
import numpy as np


def _consumers(nodes):
    result = {}
    for node in nodes:
        for input_name in node.inputs:
            result.setdefault(input_name, []).append(node)
    return result


def _output_channels(Model: TFConvertor, node) -> int | None:
    if len(node.inputs) < 2 or node.inputs[1] not in Model.params:
        return None
    weight = Model.params[node.inputs[1]]
    if not isinstance(weight, np.ndarray):
        return None
    if node.op in ("Conv", "ConvTranspose"):
        return int(weight.shape[0])
    if node.op == "MatMul" and weight.ndim == 2:
        return int(weight.shape[1])
    return None


def MergeBiasAddOnnx(Model: TFConvertor):
    """Fold only a directly-consumed channel bias Add into a weighted op.

    The old implementation searched arbitrarily far forward in node order.  A
    PiT spatial residual Add could therefore be mistaken for a Conv bias, and
    an attention relative-position Add could be attached to activation x
    activation MatMul.  Both rewrites change graph semantics.
    """

    nodes = list(Model.nodes.values())
    consumers = _consumers(nodes)
    remove_names = []

    for node in nodes:
        if node.op not in ("Conv", "ConvTranspose", "MatMul") or not node.outputs:
            continue
        # MatMul bias folding is valid only for a parameterized projection.
        if node.op == "MatMul" and (len(node.inputs) < 2 or node.inputs[1] not in Model.params):
            continue
        users = consumers.get(node.outputs[0], [])
        if len(users) != 1 or users[0].op != "Add":
            continue
        add_node = users[0]
        param_inputs = [name for name in add_node.inputs if name in Model.params]
        if len(param_inputs) != 1 or node.outputs[0] not in add_node.inputs:
            continue

        new_bias = Model.params[param_inputs[0]]
        out_channels = _output_channels(Model, node)
        if not isinstance(new_bias, np.ndarray) or out_channels is None or new_bias.size != out_channels:
            continue
        new_bias = new_bias.reshape(-1)

        old_bias = None
        if len(node.inputs) >= 3 and node.inputs[2] in Model.params:
            old_bias = Model.params[node.inputs[2]]
            if not isinstance(old_bias, np.ndarray) or old_bias.size != out_channels:
                continue
            new_bias = old_bias.reshape(-1) + new_bias

        bias_name = f"{node.name}:Bias"
        Model.params[bias_name] = np.ascontiguousarray(new_bias)
        if len(node.inputs) >= 3:
            node.inputs[2] = bias_name
        else:
            node.inputs.append(bias_name)
        node.outputs = add_node.outputs
        remove_names.append(add_node.name)

    for name in remove_names:
        Model.nodes.pop(name, None)
