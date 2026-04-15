from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def MergeMaskSoftmaxOnnx(Model:TFConvertor):
    matches = ["Mul","Add","Softmax"]
    removeNodes = []
    nodes = list(Model.nodes.values())

    def matchfunc(index):
        for one in matches:
            if index >= len(nodes):
                return False
            if nodes[index].op != one:
                return False
            index += 1
        return True

    for index,node in enumerate(nodes):
        if node.op == matches[0]:
            if matchfunc(index) is False:
                continue
            removeNodes.append(nodes[index].name)
            removeNodes.append(nodes[index+1].name)
            nodes[index+2].inputs = [nodes[index].inputs[0]]
            nodes[index+2].op = "MaskSoftmax"
            nodes[index+2].attr["upper"] = False
            nodes[index+2].attr["diagonals"] = 1
            nodes[index+2].attr["mod"] = 0



    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)

