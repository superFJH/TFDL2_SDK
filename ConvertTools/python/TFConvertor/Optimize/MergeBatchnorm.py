from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np


def MergeBatchNormOnnx(Model:TFConvertor):
    matches1 = ["Conv","BatchNormalization"]
    matches2 = ["ConvTranspose","BatchNormalization"]
    matches3 = ["MatMul","BatchNormalization"]
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
            batchnormNode = matchnodes[1]

        elif node.op == "ConvTranspose":
            matchnodes = matchfunc(index,matches2)
            if matchnodes is None:
                continue
            batchnormNode = matchnodes[1]
        elif node.op == "MatMul":
            matchnodes = matchfunc(index,matches3)
            if matchnodes is None:
                continue
            batchnormNode = matchnodes[1]
        
        convnode = node
        if convnode.inputs[1] not in Model.params.keys():
            raise RuntimeError("Merge BatchNorm Error Can't find W")
        W = Model.params[convnode.inputs[1]]
        B = None if len(convnode.inputs) < 3 else Model.params[convnode.inputs[2]]
        removeNodes.append(batchnormNode.name)
        eps = batchnormNode.attr["epsilon"]
        scale = Model.params[batchnormNode.inputs[1]]
        bias = Model.params[batchnormNode.inputs[2]]
        mean = Model.params[batchnormNode.inputs[3]]
        var = Model.params[batchnormNode.inputs[4]]
        var = np.sqrt(var+eps)
        scale = scale / var
        bias = bias - scale*mean
        if convnode.op == "MatMul":
            W = np.transpose(W,(1,0))
        assert W.shape[0] == scale.size, "conv weight mismatcht with batchnorm"
        Wshape = W.shape

        W = (W.reshape((Wshape[0],-1)) * scale.reshape((Wshape[0],1))).reshape(Wshape)
        if B is None:
            B = bias
            convnode.inputs.append(convnode.name+":Bias")
        else:
            B = B * scale + bias
        if convnode.op == "MatMul":
            W = np.transpose(W,(1,0))
        Model.params[convnode.inputs[1]] = np.ascontiguousarray(W)
        Model.params[convnode.inputs[2]] = np.ascontiguousarray(B)
        convnode.outputs = batchnormNode.outputs


    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)

def ReplaceBatchNorm(Model:TFConvertor):
    removeNodes = []
    nodes = list(Model.nodes.values())
    

    for index,node in enumerate(nodes):
        if node.op != "BatchNormalization":
            continue
        batchnormNode = node
        eps = batchnormNode.attr["epsilon"]
        scale = Model.params[batchnormNode.inputs[1]]
        bias = Model.params[batchnormNode.inputs[2]]
        mean = Model.params[batchnormNode.inputs[3]]
        var = Model.params[batchnormNode.inputs[4]]
        var = np.sqrt(var+eps)
        scale = scale / var
        bias = bias - scale*mean
        Model.params[batchnormNode.inputs[1]] = np.ascontiguousarray(scale)
        Model.params[batchnormNode.inputs[2]] = np.ascontiguousarray(bias)
        
        Model.params.pop(batchnormNode.inputs[3])
        Model.params.pop(batchnormNode.inputs[4])
        batchnormNode.inputs = [batchnormNode.inputs[0],batchnormNode.inputs[1],batchnormNode.inputs[2]]
        batchnormNode.op = "Scale"


    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)

