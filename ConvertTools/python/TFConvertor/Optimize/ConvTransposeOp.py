from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def ReshapConvTransposeWeightOnnx(Model:TFConvertor):
    nodes = list(Model.nodes.values())

    for index,node in enumerate(nodes):
        if node.op == "ConvTranspose":
            param = Model.params[node.inputs[1]]
            param = np.flip(param,(2,3))
            if node.attr["group"] == 1:
                param = np.transpose(param,(1,0,2,3))
            Model.params[node.inputs[1]] = param



