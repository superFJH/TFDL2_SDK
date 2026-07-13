from ..TFConvertor import TFConvertor
import numpy as np


def _consumers(nodes):
    result = {}
    for node in nodes:
        for input_name in node.inputs:
            result.setdefault(input_name, []).append(node)
    return result


def MergeBatchNormOnnx(Model: TFConvertor):
    nodes = list(Model.nodes.values())
    consumers = _consumers(nodes)
    remove_names = []

    for node in nodes:
        if node.op not in ("Conv", "ConvTranspose", "MatMul") or not node.outputs:
            continue
        users = consumers.get(node.outputs[0], [])
        if len(users) != 1 or users[0].op != "BatchNormalization":
            continue
        batchnorm = users[0]
        if len(node.inputs) < 2 or node.inputs[1] not in Model.params:
            continue

        weight = Model.params[node.inputs[1]]
        old_bias = Model.params[node.inputs[2]] if len(node.inputs) >= 3 and node.inputs[2] in Model.params else None
        eps = batchnorm.attr["epsilon"]
        scale = Model.params[batchnorm.inputs[1]]
        bias = Model.params[batchnorm.inputs[2]]
        mean = Model.params[batchnorm.inputs[3]]
        var = Model.params[batchnorm.inputs[4]]
        normalized_scale = scale / np.sqrt(var + eps)
        normalized_bias = bias - normalized_scale * mean

        if node.op == "MatMul":
            weight = np.transpose(weight, (1, 0))
        if weight.shape[0] != normalized_scale.size:
            continue
        weight_shape = weight.shape
        weight = (weight.reshape((weight_shape[0], -1)) * normalized_scale.reshape((-1, 1))).reshape(weight_shape)
        if old_bias is None:
            fused_bias = normalized_bias
        else:
            fused_bias = old_bias * normalized_scale + normalized_bias
        if node.op == "MatMul":
            weight = np.transpose(weight, (1, 0))

        # Never mutate or delete tied initializers: onnxslim deliberately ties
        # identical BN parameters across many LeViT blocks.
        weight_name = f"{node.name}:BatchNormWeight"
        bias_name = f"{node.name}:BatchNormBias"
        Model.params[weight_name] = np.ascontiguousarray(weight)
        Model.params[bias_name] = np.ascontiguousarray(fused_bias)
        node.inputs[1] = weight_name
        if len(node.inputs) >= 3:
            node.inputs[2] = bias_name
        else:
            node.inputs.append(bias_name)
        node.outputs = batchnorm.outputs
        remove_names.append(batchnorm.name)

    for name in remove_names:
        Model.nodes.pop(name, None)


def ReplaceBatchNorm(Model: TFConvertor):
    for batchnorm in list(Model.nodes.values()):
        if batchnorm.op != "BatchNormalization":
            continue
        eps = batchnorm.attr["epsilon"]
        scale = Model.params[batchnorm.inputs[1]]
        bias = Model.params[batchnorm.inputs[2]]
        mean = Model.params[batchnorm.inputs[3]]
        var = Model.params[batchnorm.inputs[4]]
        normalized_scale = scale / np.sqrt(var + eps)
        normalized_bias = bias - normalized_scale * mean

        # Give every Scale private parameters. Shared/tied source tensors must
        # not be overwritten by the first BatchNorm that consumes them.
        scale_name = f"{batchnorm.name}:Scale"
        bias_name = f"{batchnorm.name}:Bias"
        Model.params[scale_name] = np.ascontiguousarray(normalized_scale)
        Model.params[bias_name] = np.ascontiguousarray(normalized_bias)
        batchnorm.inputs = [batchnorm.inputs[0], scale_name, bias_name]
        batchnorm.op = "Scale"
