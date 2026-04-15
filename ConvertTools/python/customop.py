from TFConvertor import OnnxConvertor,TFConvertor,OnnxNodeConvertor,NodeConvertor
from TFDL2 import TFDataType,Op,DECODER_FLAGS
from collections import OrderedDict
import numpy as np
import glob
import os
from typing import Dict, List, Union

def PillarScatterPlugin(*args,node:NodeConvertor):
    import json
    featureY,featureX = node.attr['dense_shape']
    out = Op.Custom(args,(node.outputs[0],),"PPScatterPlugin",json.dumps({
        "featureY":featureY,
        "featureX":featureX,
    }
    ))

    return out[0]
    
def Optimize1(*args,node:NodeConvertor):
    a = Op.Transpose(args[0],(0,2,1))
    a = Op.ExpandDims(a,(-1,))
    a = Op.Convolution2(a,args[1],args[2],1,0,1,1,64,1)
    a = Op.ReLU(a)
    a = Op.Squeeze(a,(3,))
    a = Op.Reshape(a,(10,1000,64,-1))
    #a = Op.ExpandDims(args[0],(0,))
    a = Op.ReduceMax(a,(3,),keep_dims=True)
    #a = Op.Squeeze(a,(0,))
    a = Op.Reshape(a,(-1,64))


    return a

def custom1(Model:TFConvertor):

    #Model.nodes.pop("/Relu")

    #Model.nodes['/Squeeze'].inputs = ["/depth_head/output_conv2/output_conv2.3/Relu_output_0"]
    Model.nodes['ConvTranspose_243'].op = "Conv"

def custom2(Model:TFConvertor):
    nodes = list(Model.nodes.values())

    for index,node in enumerate(nodes):
        
        if node.op == "TransForm":
            largeConvW = None
            largeConvWname = None
            largeConvB = None
            largeConvBname = None
            for ii,iname in enumerate(node.inputs):
                if iname in Model.params:
                    param = Model.params[iname]
                    if len(param.shape) == 4:
                        largeConvW = param
                        largeConvWname = iname
                    elif largeConvW is not None and len(param.shape) == 1:
                        largeConvB = param
                        largeConvBname = iname
                
                if largeConvW is not None and largeConvB is not None:
                    break
        
            outchannel = largeConvW.shape[0]
            splitchannel = outchannel//3
            for i in range(3):
                wname = "{0}_w_{1}".format(largeConvWname,i)
                bname = "{0}_b_{1}".format(largeConvBname,i)
                node.inputs.append(wname)
                Model.params[wname] = largeConvW[i*splitchannel:(i+1)*splitchannel]
                node.inputs.append(bname)
                Model.params[bname] = largeConvB[i*splitchannel:(i+1)*splitchannel]
        

def custom3(Model:TFConvertor):
    
    matmulnode = Model.nodes['MatMul_161']
    batchnormnode = Model.nodes['BatchNormalization_163']
    W = Model.params[matmulnode.inputs[1]]
    scale = Model.params[batchnormnode.inputs[1]]
    bias = Model.params[batchnormnode.inputs[2]]
    W = W*scale
    W = np.transpose(W,(1,0))
    W = W[:,:,np.newaxis,np.newaxis]
    B = bias
    Model.params[matmulnode.inputs[1]] = np.ascontiguousarray(W)
    matmulnode.inputs.append(batchnormnode.inputs[2])
    Model.nodes.pop(batchnormnode.name)
    Model.nodes.pop("Transpose_162")
    Model.nodes.pop("Transpose_164")
    Model.nodes.pop("Relu_165")
    Model.nodes.pop("ReduceMax_166")
    '''
    Model.nodes['Relu_165'].inputs = [matmulnode.outputs[0]]
    addnode = OnnxNodeConvertor(Model)
    addnode.op = "Transpose"
    addnode.inputs = [matmulnode.inputs[0]]
    addnode.outputs = [matmulnode.inputs[0]+"_transpose"]
    addnode.name = matmulnode.inputs[0]+"_transpose"
    addnode.attr["perm"] = [0,2,1]
    matmulnode.inputs[0] = matmulnode.inputs[0]+"_transpose"
    matmulnode.inputs = [matmulnode.inputs[1],matmulnode.inputs[0],matmulnode.inputs[2]]
    #matmulnode.op = "Conv"
    #matmulnode.attr["kernel_shape"] = [1,1]
    #matmulnode.attr["dilations"] = [1,1]
    #matmulnode.attr["strides"] = [1,1]
    #matmulnode.attr["pads"] = [0,0,0,0]
    #matmulnode.attr["group"] = 1
    Model.nodes[matmulnode.inputs[0]+"_transpose"] = addnode

    Model.nodes['ReduceMax_166'].attr['axes'] = [2]
    Model.nodes['ReduceMax_166'].attr['keepdims'] = False
    '''
    addnode = OnnxNodeConvertor(Model)
    addnode.op = "Optimize1"
    addnode.inputs = matmulnode.inputs
    addnode.outputs = ["333"]
    addnode.name = "Opimize1"
    Model.nodes[addnode.name] = addnode
    Model.nodes.pop(matmulnode.name)
    

    

                    
