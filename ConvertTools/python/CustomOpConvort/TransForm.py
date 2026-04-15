from TFConvertor import OnnxConvertor,TFConvertor,OnnxNodeConvertor,NodeConvertor
import argparse
from TFDL2 import TFDataType,DECODER_FLAGS
from TFDL2.Op import *


def TransFormParser(cls,*args,node:NodeConvertor):
        
            
    outchannel = node.graph.params[str(args[6])].shape[0]
    querySize = node.graph.params[str(args[6])].shape[1]
    reshapeinfo = node.graph.params[str(args[8])]

    
    getin = LayerNorm2(args[0],args[3],args[4])
    getin = Transpose(getin,(0,2,1))
    getin = Reshape(getin,(0,querySize,-1,64))
    '''
    a = Convolution2(getin,args[6],args[7],1,0,1,1,outchannel,1)#MatMul2(args[3],getin,args[4],bias2Row=True)
    
    a = Reshape(a,(0,outchannel,-1))
    
    a = Reshape(a,(0,3*reshapeinfo[-2],reshapeinfo[-1],-1))
    a = Transpose(a,(0,1,3,2))
    a = Reshape(a,(0,3,reshapeinfo[-2],-1,reshapeinfo[-1]))
    a1,a2,a3 = Split(a,1,3) # b,16,64,257
    a1 = Squeeze(a1,(1,))
    a2 = Squeeze(a2,(1,))
    a3 = Squeeze(a3,(1,))
    '''
    a1 = Convolution2(getin,args[27],args[28],1,0,1,1,outchannel//3,1)
    a1 = Reshape(a1,(0,reshapeinfo[-2],reshapeinfo[-1],-1))
    a1 = Transpose(a1,(0,1,3,2))
    a2 = Convolution2(getin,args[29],args[30],1,0,1,1,outchannel//3,1)
    a2 = Reshape(a2,(0,reshapeinfo[-2],reshapeinfo[-1],-1))
    #a2 = Transpose(a2,(0,1,3,2))
    a3 = Convolution2(getin,args[31],args[32],1,0,1,1,outchannel//3,1)
    a3 = Reshape(a3,(0,reshapeinfo[-2],reshapeinfo[-1],-1))
    a3 = Transpose(a3,(0,1,3,2))
    #a1 = Reshape(a1,(12,64,-1))
    #a2 = Reshape(a2,(12,64,-1))
    #a3 = Reshape(a3,(12,64,-1))
    #a2 = Transpose(a2,(0,1,3,2)) # 16,257,64
    a1 = a1 * (1.0/np.sqrt(reshapeinfo[-1]))
    a0 = MatMul(a1,a2,hasBias=False,bias2Row=False)
    #a0 = a0 * (1.0/np.sqrt(reshapeinfo[-1]))
    #a0 = MaskSoftmax(a0,-1,False,1,0)
    a0 = Softmax(a0,-1)
    #a0 = Transpose(a0,(0,1,3,2))#
    a = MatMul(a0,a3,hasBias=False,bias2Row=False)
    a = Transpose(a,(0,1,3,2))
    a = Reshape(a,(0,querySize,-1,64))
    #outchannel = node.graph.params[str(args[12])].shape[0]
    a = Convolution2(a,args[13],args[14],1,0,1,1,querySize,1)#MatMul2(args[5],a,args[6],bias2Row=True)
    a = Reshape(a,(0,querySize,-1))
    a = Transpose(a,(0,2,1))
    #a = a * args[13] # LayerScale
    #a = Reshape(a,(1,-1,768))
    bert1 =  a + args[0]
    getin = LayerNorm2(bert1,args[16],args[17])
    a = Transpose(getin,(0,2,1))
    a = Reshape(a,(0,querySize,-1,64))
    outchannel = node.graph.params[str(args[18])].shape[0]
    a = Convolution2(a,args[18],args[19],1,0,1,1,outchannel,1)#MatMul2(args[3],a,args[4],bias2Row=True)
    a = GeLU(a)
    #a = Swish(a)
    outchannel = node.graph.params[str(args[25])].shape[0]
    a = Convolution2(a,args[25],args[26],1,0,1,1,outchannel,1)#MatMul2(args[5],a,args[6],bias2Row=True)
    a = Reshape(a,(0,outchannel,-1))
    a = Transpose(a,(0,2,1))
    #a = a * args[23] # LayerScale
    out =  a + bert1
    if len(node.outputs) > 1:
        return [args[0],out]
    else:
        return out


