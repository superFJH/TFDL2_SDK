from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def MergeBiasAddOnnx(Model:TFConvertor):
    matches1 = ["Conv","Add"]
    matches2 = ["ConvTranspose","Add"]
    matches3 = ["MatMul","Add"]
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
        if node.op == "Conv" or node.op == "ConvTranspose" or node.op == "MatMul":
            pass
        else:
            continue

        if node.op == "Conv":
            matchnodes = matchfunc(index,matches1)
            if matchnodes is None:
                continue
            biasaddNode = matchnodes[1]

        elif node.op == "ConvTranspose":
            matchnodes = matchfunc(index,matches2)
            if matchnodes is None:
                continue
            biasaddNode = matchnodes[1]
        elif node.op == "MatMul":
            matchnodes = matchfunc(index,matches3)
            if matchnodes is None:
                continue
            biasaddNode = matchnodes[1]
        #biasaddNode = nodes[index+1]
        
        if biasaddNode.inputs[0] not in Model.params and biasaddNode.inputs[1] not in Model.params:
            continue

        if biasaddNode.inputs[0] in Model.params:
            newbias = Model.params[biasaddNode.inputs[0]]
        if biasaddNode.inputs[1] in Model.params:
            newbias = Model.params[biasaddNode.inputs[1]]

        

        convnode = node
        B = None if len(convnode.inputs) < 3 else Model.params[convnode.inputs[2]]

        if B is None:
            B = newbias
            convnode.inputs.append(convnode.name+":Bias")
        else:
            B = B  + newbias
        B = B.flatten()
        Model.params[convnode.inputs[2]] = np.ascontiguousarray(B)
        convnode.outputs = biasaddNode.outputs
        biasaddNode.inputs = []
        removeNodes.append(biasaddNode.name)


    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)

