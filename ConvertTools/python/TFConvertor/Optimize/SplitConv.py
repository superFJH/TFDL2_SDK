from ..TFConvertor import NodeConvertor,TFConvertor

import numpy as np
import copy

def SplitConvOnnx(Model:TFConvertor):
    from ..onnxConvertor import OnnxNodeConvertor
    matches = [set(["Conv",'ConvTranspose']),set(["Swish",'HardSwish','ReLU','Mish','HardSigmoid',"Sigmoid",'Relu6','LeakyRelu',"GeLU",'Tanh','Split']),"Split"]
    removeNodes = []
    addNodes = []
    nodes = list(Model.nodes.values())

    def matchfunc(index,matches):
        lastoutput = None
        matchnodes = []
        for one in matches:
            found = False
            for i in range(index,len(nodes),1):
                if isinstance(one, set):
                    if nodes[i].op in one and (lastoutput is None or lastoutput in nodes[i].inputs):
                        found = True
                        index = i + 1
                        lastoutput = nodes[i].outputs[0]
                        matchnodes.append(nodes[i])
                        if nodes[i].op == 'Split':
                            return matchnodes
                        break
                else:
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
        if node.op in matches[0]:
            matchnodes = matchfunc(index,matches)
            if matchnodes is None:
                continue
            splitnode = matchnodes[-1]

            convnode = matchnodes[0]
            actnode = matchnodes[1] if len(matchnodes) > 2 else None
            if splitnode.attr['axis'] != 1:
                continue
            
            if 'split' not in splitnode.attr:
                continue
            else:
                removeNodes.extend([node.name for node in matchnodes])
                st = 0
                for index,channels in enumerate(splitnode.attr['split']):
                    addnode = OnnxNodeConvertor(graph=Model)
                    addnode.op = "Conv"
                    addnode.name = f"{convnode.name}_{index}"
                    addnode.attr = copy.deepcopy(matchnodes[0].attr)
                    addnode.inputs = copy.deepcopy(convnode.inputs)
                    addnode.inputs[1] = f"{convnode.inputs[1]}_{index}"
                    
                    
                    
                    param = Model.params[convnode.inputs[1]]
                    thisparam = param[st:st+channels].copy()
                    Model.params[f"{convnode.inputs[1]}_{index}"] = np.ascontiguousarray(thisparam)

                    if len(addnode.inputs) > 2:
                        addnode.inputs[2] = f"{convnode.inputs[2]}_{index}"
                        param = Model.params[convnode.inputs[2]]
                        thisparam = param[st:st+channels].copy()
                        Model.params[f"{convnode.inputs[2]}_{index}"] = np.ascontiguousarray(thisparam)



                    if actnode is None:
                        addnode.outputs = [splitnode.outputs[index]]
                    else:
                        addnode.outputs = [addnode.name]
                        addnode2 = OnnxNodeConvertor(graph=Model)
                        addnode2.op = actnode.op
                        addnode2.name = f"{actnode.name}_{index}"
                        addnode2.attr = copy.deepcopy(actnode.attr)
                        addnode2.inputs = [addnode.name]
                        addnode2.outputs = [splitnode.outputs[index]]

                    addNodes.append(addnode) 
                    if actnode is not None:
                        addNodes.append(addnode2)
                    
                    st += channels




    for node in removeNodes:
        for iname in Model.nodes[node].inputs:
            if iname in Model.params:
                del Model.params[iname]
        Model.nodes.pop(node)
    
    for node in addNodes:
        Model.nodes[node.name] = node

