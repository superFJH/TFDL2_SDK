from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def MergeGeLUOnnx(Model:TFConvertor):
    matches = ["Div","Erf","Add","Mul","Mul"]
    removeNodes = []
    nodes = list(Model.nodes.values())

    def matchfunc(index,matches):
        lastoutput = None
        matchnodes = []
        for one in matches:
            found = False
            for i in range(index,len(nodes),1):
                if nodes[i].op == one and (lastoutput is None or lastoutput in nodes[i].inputs):
                    found = True
                    index = i + 1
                    lastoutput = nodes[i].outputs[0]
                    matchnodes.append(nodes[i])
                    break
            if found is False:
                return None
        return matchnodes

    for index,node in enumerate(nodes):
        if node.op == matches[0]:
            matchnodes = matchfunc(index,matches)
            if matchnodes is None:
                continue
            removeNodes = [node.name for node in matchnodes[:-1]]
            matchnodes[4].inputs = [matchnodes[0].inputs[0]]
            matchnodes[4].op = "GeLU"



    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)

