from ..TFConvertor import NodeConvertor,TFConvertor
import numpy as np
import copy

def MergeBertOnnx(Model:TFConvertor):
    removeNodes = []
    nodes = list(Model.nodes.values())
    Found = False

    allinputs = []
    tmpparam = {}
    tmpparam2 = {}
    mergenodes = []
    for index,node in enumerate(nodes):
        if node.op == "LayerNorm":
            if Found is False:
                Found = True
                allinputs.extend(node.inputs)
                #node.inputs = []
                mergenodes.append(node.name)
            else:
                if len(mergenodes)  < 5:
                    mergenodes = []
                    tmpparam.clear()
                    allinputs.clear()
                    tmpparam2.clear()
                    allinputs.extend(node.inputs)

                    mergenodes.append(node.name)
                    continue
                Found = False
                mergenodes.pop()
                removeNodes.extend(mergenodes)
                mergenodes = []
                for k,v in tmpparam.items():
                    Model.params[k] = v

                for k,v in tmpparam2.items():
                    del Model.params[k]
                tmpparam.clear()
                tmpparam2.clear()
                nodes[index-1].op = "Bert"
                nodes[index-1].inputs = copy.deepcopy(allinputs)
                allinputs.clear()
        else:
            if Found is True:
                mergenodes.append(node.name)
                for iname in node.inputs:
                    if iname in Model.params:
                        if type(Model.params[iname]) is np.ndarray:
                            pp = Model.params[iname]
                            if len(pp.shape) == 2:
                                pp = np.transpose(pp,(1,0))
                                pp = pp[:,:,np.newaxis,np.newaxis]
                                tmpparam[iname] = np.ascontiguousarray(pp)
                                #Model.params[iname] = pp
                            allinputs.append(iname)
                        else:
                            tmpparam2[iname] = Model.params[iname]

    for node in removeNodes:
        Model.nodes.pop(node)

    removeNodes = []
    nodes = list(Model.nodes.values())
    Found = False

    allinputs = []
    tmpparam = {}
    tmpparam2 = {}
    mergenodes = []
    for index,node in enumerate(nodes):
        if node.op == "Bert":
            if Found is False:
                Found = True
                allinputs.extend(node.outputs)
            else:
                mergenodes.pop()
                removeNodes.extend(mergenodes)
                mergenodes = []
                for k,v in tmpparam.items():
                    Model.params[k] = v

                for k,v in tmpparam2.items():
                    del Model.params[k]
                tmpparam.clear()
                tmpparam2.clear()
                nodes[index-1].op = "Bert2"
                nodes[index-1].inputs = copy.deepcopy(allinputs)
                allinputs.clear()
                allinputs.extend(node.outputs)
        else:
            if Found is True:
                if node.op == "Transpose":
                    removeNodes.extend(mergenodes)
                    mergenodes = []
                    for k,v in tmpparam.items():
                        Model.params[k] = v

                    for k,v in tmpparam2.items():
                        del Model.params[k]
                    tmpparam.clear()
                    tmpparam2.clear()
                    nodes[index].op = "Bert2"
                    nodes[index].inputs = copy.deepcopy(allinputs)
                    allinputs.clear()
                    break
                mergenodes.append(node.name)
                for iname in node.inputs:
                    if iname in Model.params:
                        if type(Model.params[iname]) is np.ndarray:
                            pp = Model.params[iname]
                            if len(pp.shape) > 1:
                                pp = np.transpose(pp,(1,0))
                                pp = pp[:,:,np.newaxis,np.newaxis]
                                tmpparam[iname] = np.ascontiguousarray(pp)
                                #Model.params[iname] = pp
                            allinputs.append(iname)
                        else:
                            tmpparam2[iname] = Model.params[iname]

    for node in removeNodes:
        Model.nodes.pop(node)

def MergeConvBertOnnx(Model:TFConvertor):
    removeNodes = []
    nodes = list(Model.nodes.values())
    Found = False

    allinputs = []
    tmpparam = {}
    tmpparam2 = {}
    mergenodes = []
    for index,node in enumerate(nodes):
        if node.op == "LayerNorm":
            if Found is False:
                Found = True
                allinputs.extend(node.inputs)
                #node.inputs = []
                mergenodes.append(node.name)
            else:
                if len(mergenodes)  < 5:
                    mergenodes = []
                    tmpparam.clear()
                    allinputs.clear()
                    tmpparam2.clear()
                    allinputs.extend(node.inputs)

                    mergenodes.append(node.name)
                    continue
                Found = False
                mergenodes.pop()
                removeNodes.extend(mergenodes)
                mergenodes = []
                for k,v in tmpparam.items():
                    Model.params[k] = v

                for k,v in tmpparam2.items():
                    del Model.params[k]
                tmpparam.clear()
                tmpparam2.clear()
                nodes[index-1].op = "ConvBert"
                nodes[index-1].inputs = copy.deepcopy(allinputs)
                allinputs.clear()
        else:
            if Found is True:
                mergenodes.append(node.name)
                for iname in node.inputs:
                    if iname in Model.params:
                        if type(Model.params[iname]) is np.ndarray:
                            pp = Model.params[iname]
                            if len(pp.shape) > 1 and len(pp.shape) < 4:
                                pp = np.transpose(pp,(1,0))
                                pp = pp[:,:,np.newaxis,np.newaxis]
                                tmpparam[iname] = np.ascontiguousarray(pp)
                                #Model.params[iname] = pp
                            allinputs.append(iname)
                        else:
                            tmpparam2[iname] = Model.params[iname]

    for node in removeNodes:
        Model.nodes.pop(node)

    removeNodes = []
    nodes = list(Model.nodes.values())
    Found = False

    allinputs = []
    tmpparam = {}
    tmpparam2 = {}
    mergenodes = []
    for index,node in enumerate(nodes):
        if node.op == "ConvBert":
            if Found is False:
                Found = True
                allinputs.extend(node.outputs)
            else:
                mergenodes.pop()
                removeNodes.extend(mergenodes)
                mergenodes = []
                for k,v in tmpparam.items():
                    Model.params[k] = v

                for k,v in tmpparam2.items():
                    del Model.params[k]
                tmpparam.clear()
                tmpparam2.clear()
                nodes[index-1].op = "ConvBert2"
                nodes[index-1].inputs = copy.deepcopy(allinputs)
                allinputs.clear()
                allinputs.extend(node.outputs)
        else:
            if Found is True:
                if node.op == "Transpose":
                    removeNodes.extend(mergenodes)
                    mergenodes = []
                    for k,v in tmpparam.items():
                        Model.params[k] = v

                    for k,v in tmpparam2.items():
                        del Model.params[k]
                    tmpparam.clear()
                    tmpparam2.clear()
                    nodes[index].op = "ConvBert2"
                    nodes[index].inputs = copy.deepcopy(allinputs)
                    allinputs.clear()
                    break
                mergenodes.append(node.name)
                for iname in node.inputs:
                    if iname in Model.params:
                        if type(Model.params[iname]) is np.ndarray:
                            pp = Model.params[iname]
                            if len(pp.shape) > 1 and len(pp.shape) < 4:
                                pp = np.transpose(pp,(1,0))
                                pp = pp[:,:,np.newaxis,np.newaxis]
                                tmpparam[iname] = np.ascontiguousarray(pp)
                                #Model.params[iname] = pp
                            allinputs.append(iname)
                        else:
                            tmpparam2[iname] = Model.params[iname]

    for node in removeNodes:
        Model.nodes.pop(node)



