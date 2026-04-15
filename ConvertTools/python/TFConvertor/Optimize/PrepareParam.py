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
            Model.params[node.inputs[1]] = np.ascontiguousarray(param)
        
        if node.op == "TransForm":
            for ii,iname in enumerate(node.inputs):
                if iname in Model.params:
                    param = Model.params[iname]
                    if len(param.shape) == 2:
                        param = np.transpose(param,(1,0))
                        param = param[:,:,np.newaxis,np.newaxis]
                        Model.params[iname] = np.ascontiguousarray(param)

def RemoveCastOnnx(Model:TFConvertor):
    nodes = list(Model.nodes.values())
    removeNodes = []
    for index,node in enumerate(nodes):
        if node.op == "Cast":
            if nodes[index-1].op == "Cast":
                continue
            if nodes[index-1].outputs != node.inputs:
                continue
            nodes[index-1].outputs = node.outputs
            removeNodes.append(node.name)


    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)

def ReshapGemmWeightOnnx(Model:TFConvertor):
    nodes = list(Model.nodes.values())

    for index,node in enumerate(nodes):
        if node.op == "Gemm":
            alpha = node.attr["alpha"] if "alpha" in node.attr else 1.0
            beta = node.attr["beta"] if "beta" in node.attr else 1.0
            tranA = node.attr["transA"]==1 if "transA" in node.attr else False
            tranB = node.attr["transB"]==1 if "transB" in node.attr else False
            param = Model.params[node.inputs[1]]
            if not tranB:
                param = np.transpose(param,(1,0))
                node.attr["transB"] = 0
            param = param * alpha
            Model.params[node.inputs[1]] = np.ascontiguousarray(param)
            if len(node.inputs) > 2:
                Model.params[node.inputs[2]] = np.ascontiguousarray(Model.params[node.inputs[2]]*beta)


