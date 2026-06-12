from . import TFDL2
from .TFDL2 import TFSymbol
from .Common import TFDataType,DECODER_FLAGS
import numpy as np 

def Add(InputA ,InputB ,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]

    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.Add(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.Add1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.Add2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")


def Sub(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.Sub(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.Sub1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.Sub2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")

def Mul(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.Mul(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.Mul1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.Mul2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")

def Div(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.Div(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.Div1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.Div2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")


def GT(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.GT(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.GT1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.GT2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")

def GE(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.GE(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.GE1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.GE2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")

def LT(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.LT(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.LT1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.LT2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")

def LE(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.LE(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.LE1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.LE2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")

def EQ(InputA,InputB,dtype=TFDataType.TFDL_FLOAT.value)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    if type(InputA) is TFSymbol and type(InputB) is TFSymbol:
        return TFDL2.EQ(InputA,InputB)
    elif type(InputA) is TFSymbol and type(InputB) is not TFSymbol:
        assert(dtype!=None)
        return TFDL2.EQ1(InputA,InputB,dtype)
    elif type(InputA) is not TFSymbol and type(InputB) is TFSymbol:
        assert(dtype!=None)
        return TFDL2.EQ2(InputA,dtype,InputB)
    else:
        raise RuntimeError("input type Wrong")


def Min(InputA : TFSymbol,InputB :TFSymbol)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    return TFDL2.Min(InputA,InputB)

def Max(InputA : TFSymbol,InputB :TFSymbol)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    return TFDL2.Max(InputA,InputB)

def Convolution(Input: TFSymbol,kernel,pad,stride,dilation:int,outChannel:int,group:int,hasBias:bool=True)->TFSymbol:
    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")

    return TFDL2.Convolution(Input,kernel,pad,stride,dilation,outChannel,group,hasBias)

def Scale(Input: TFSymbol,hasBias:bool=True)->TFSymbol:
    return TFDL2.Scale(Input,hasBias)

def InnerProduct(Input: TFSymbol,outchannels:int,hasBias:bool=True)->TFSymbol:
    return TFDL2.InnerProduct(Input,hasBias,outchannels)

def Convolution2(Input: TFSymbol,weight: TFSymbol,bias: TFSymbol,kernel,pad,stride,dilation:int,outChannel:int,group:int)->TFSymbol:
    if weight is None:
        weight = TFSymbol()
    if bias is None:
        bias = TFSymbol()

    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")

    return TFDL2.Convolution2(Input,weight,bias,kernel,pad,stride,dilation,outChannel,group)

def Scale2(Input: TFSymbol,weight: TFSymbol,bias: TFSymbol)->TFSymbol:
    if weight is None:
        weight = TFSymbol()
    if bias is None:
        bias = TFSymbol()

    return TFDL2.Scale2(Input,weight,bias)

def InnerProduct2(Input: TFSymbol,weight: TFSymbol,bias: TFSymbol,outchannels:int)->TFSymbol:
    if weight is None:
        weight = TFSymbol()
    if bias is None:
        bias = TFSymbol()

    return TFDL2.InnerProduct2(Input,weight,bias,outchannels)

def TFConvolution(Input: TFSymbol,kernel,Same:bool,stride,dilation:int,outChannel:int,group:int,hasBias:bool=True)->TFSymbol:
    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")
    return TFDL2.TFConvolution(Input,kernel,Same,stride,dilation,outChannel,group,hasBias)

def DeConvolution(Input: TFSymbol,kernel,pad,stride,dilation:int,outChannel:int,group:int,hasBias:bool=True,outPadH:int=0,outPadW:int=0)->TFSymbol:
    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")
    return TFDL2.DeConvolution(Input,kernel,pad,stride,dilation,outChannel,group,hasBias,outPadH,outPadW)

def DeConvolution2(Input: TFSymbol,weight: TFSymbol,bias: TFSymbol,kernel,pad,stride,dilation:int,outChannel:int,group:int,outPadH:int=0,outPadW:int=0)->TFSymbol:
    if weight is None:
        weight = TFSymbol()
    if bias is None:
        bias = TFSymbol()

    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")
    return TFDL2.DeConvolution2(Input,weight,bias,kernel,pad,stride,dilation,outChannel,group,outPadH,outPadW)

def GlobalAvePooling(Input:TFSymbol):
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.GlobalAvePooling(Input)

def MaxPooling(Input:TFSymbol,kernel,pad,stride,ceilmode:bool)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")
    return TFDL2.MaxPooling(Input,kernel,pad,stride,ceilmode)

def AvePooling(Input:TFSymbol,kernel,pad,stride,ceilmode:bool)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")
    return TFDL2.AvePooling(Input,kernel,pad,stride,ceilmode)

def TFMaxPooling(Input:TFSymbol,kernel:tuple,stride:tuple,Same:bool)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]

    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")

    return TFDL2.TFMaxPooling(Input,kernel,Same,stride)

def TFAvePooling(Input:TFSymbol,kernel:tuple,stride:tuple,Same:bool)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]

    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")

    return TFDL2.TFAvePooling(Input,kernel,Same,stride)

def MatMul(InputA:TFSymbol,InputB:TFSymbol,transA:bool=False,transB:bool=False,hasBias:bool=False,bias2Row:bool=False)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]
    return TFDL2.MatMul(InputA,InputB,transA,transB,hasBias,bias2Row)

def MatMulWeight(InputA:TFSymbol,dimm:int,dimk:int,transA:bool=False,transB:bool=False,hasBias:bool=False,bias2Row:bool=False)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    return TFDL2.MatMulWeight(InputA,dimm,dimk,transA,transB,hasBias,bias2Row)

def MatMul2(InputA:TFSymbol,InputB:TFSymbol,Bias:TFSymbol,transA:bool=False,transB:bool=False,bias2Row:bool=False)->TFSymbol:
    if type(InputA) == tuple:
        assert(len(InputA)==1)
        InputA = InputA[0]
    if type(InputB) == tuple:
        assert(len(InputB)==1)
        InputB = InputB[0]

    if Bias is None:
        Bias = TFSymbol()

    return TFDL2.MatMul2(InputA,InputB,Bias,transA,transB,bias2Row)


def ReLU(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ReLU(Input)

def ELU(Input:TFSymbol,alpha:float)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ELU(Input,alpha)

def Mish(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Mish(Input)

def ReLUX(Input:TFSymbol,threashold:float=0.0)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ReLUX(Input,threashold)

def LeakyReLU(Input:TFSymbol,negative_slop:float=0.0)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.LeakyReLU(Input,negative_slop)

def PReLU(Input:TFSymbol,Weight:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.PReLU(Input,Weight)

def Swish(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Swish(Input)

def HardSwish(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.HardSwish(Input)

def Tanh(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Tanh(Input)

def GeLU(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.GeLU(Input)

def Sigmoid(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Sigmoid(Input)

def HardSigmoid(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.HardSigmoid(Input)

def HardSigmoid2(Input:TFSymbol,alpha:float,beta:float)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.HardSigmoid2(Input,alpha,beta)

def Exp(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Exp(Input)

def Softmax(Input:TFSymbol,axis:int)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Softmax(Input,axis)

def Reshape(Input:TFSymbol,shape:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Reshape(Input,shape)

def Reshape2(Input:TFSymbol,shape:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Reshape2(Input,shape)

def BroadCast(Input:TFSymbol,shape:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.BroadCast(Input,shape)

def BroadCast2(Input:TFSymbol,shape:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.BroadCast2(Input,shape)

def Squeeze(Input:TFSymbol,dims:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Squeeze(Input,dims)

def Pad(Input:TFSymbol,pad:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Pad(Input,pad)

def ZeroMask(Input:TFSymbol,mask:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ZeroMask(Input,mask)

def Pad2(Input:TFSymbol,pad:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Pad2(Input,pad)

def Flatten(Input:TFSymbol,start_axis:int=0,end_axis=-1)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Flatten(Input,start_axis,end_axis)

def Flatten2Matrix(Input:TFSymbol,axis:int=0)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Flatten2Matrix(Input,axis)

def Transpose(Input:TFSymbol,dims:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Transpose(Input,dims)

def ReduceMax(Input:TFSymbol,dims:tuple,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ReduceMax(Input,dims,keep_dims)

def ReduceMean(Input:TFSymbol,dims:tuple,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ReduceMean(Input,dims,keep_dims)

def ReduceSum(Input:TFSymbol,dims:tuple,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ReduceSum(Input,dims,keep_dims)

def ReduceMin(Input:TFSymbol,dims:tuple,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ReduceMin(Input,dims,keep_dims)

def Mean(Input:TFSymbol,dims:tuple,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.MeanOp(Input,dims,keep_dims)

def Concat(Inputs:tuple,axis:int)->TFSymbol:
    return TFDL2.Concat(Inputs,axis)

def Gather(Input:TFSymbol,indices:TFSymbol,axis:int)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Gather(Input,indices,axis)

def Correlation(Input1:TFSymbol,Input2:TFSymbol,kernel_size:int,pad_size:int,max_displacement:int,stride1:int,stride2:int)->TFSymbol:
    return TFDL2.Correlation(Input1,Input2,kernel_size,pad_size,max_displacement,stride1,stride2)

def Gather2(Input:TFSymbol,indices,axis:int)->TFSymbol:
    if isinstance(indices,tuple):
        pass
    elif isinstance(indices,list):
        indices = tuple(indices)
    elif isinstance(indices,int):
        indices = (indices,)
    elif isinstance(indices,float):
        indices = (int(indices),)
    else:
        raise RuntimeError("Error type for indices");
    return TFDL2.Gather2(Input,indices,axis)

def Split(Input:TFSymbol,axis:int,split:int)->list:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Split(Input,axis,split)

def Einsum(eq:str,inputs)->list:
    if type(inputs) == tuple:
        return TFDL2.Einsum(eq,inputs)
    elif type(inputs) == TFSymbol:
        return TFDL2.Einsum(eq,(inputs,))
    else:
        raise RuntimeError("Error type for inputs")

def Slice(Input:TFSymbol,axis:int,split:tuple)->list:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Slice(Input,axis,split)

def Flip(Input:TFSymbol,dims:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Flip(Input,dims)

def Reshape(Input:TFSymbol,shape:tuple)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Reshape(Input,shape)

def Quantize(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Quantize(Input)

def DeQuantize(Input:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.DeQuantize(Input)

def ReadImg(Context,imgLen:int,scale:tuple, mean:tuple,shape:tuple,outDatatype:TFDataType,decoderFlags:DECODER_FLAGS)->TFSymbol:
    return TFDL2.ReadImg(Context,imgLen,scale,mean,shape,outDatatype.value,decoderFlags)

def Placeholder(Context,batch:int,shape:tuple,outDatatype:TFDataType,scale:tuple = (), mean:tuple = ())->TFSymbol:
    return TFDL2.Placeholder(Context,batch,scale,mean,shape,outDatatype.value)

def Placeholder2(Context,shape:tuple,outDatatype:TFDataType,scale:tuple = (), mean:tuple = ())->TFSymbol:
    return TFDL2.Placeholder2(Context,scale,mean,shape,outDatatype.value)

def BilnearReSize1(Input:TFSymbol,outheight:int,outwidth:int,align_corners:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.BilnearReSize1(Input,outheight,outwidth,align_corners)

def BilnearReSize2(Input:TFSymbol,scale:float,align_corners:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.BilnearReSize2(Input,scale,align_corners)

def BilnearReSize3(Input:TFSymbol,shape:TFSymbol,align_corners:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.BilnearReSize3(Input,shape,align_corners)

def NearestReSize1(Input:TFSymbol,outheight:int,outwidth:int)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.NearestReSize1(Input,outheight,outwidth)

def NearestReSize2(Input:TFSymbol,scale:float)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.NearestReSize2(Input,scale)

def NearestReSize3(Input:TFSymbol,shape:TFSymbol)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.NearestReSize3(Input,shape)

def Upsample(Input:TFSymbol,scale:int)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.Upsample(Input,scale)

def CropAndResize(Input:TFSymbol,AlignMask:TFSymbol,outheight:int,outwidth:int)->TFSymbol:
    if type(Input) == tuple or type(Input) == list:
        assert(len(Input)==1)
        Input = Input[0]

    if type(AlignMask) == tuple or type(AlignMask) == list:
        assert(len(AlignMask)==1)
        AlignMask = AlignMask[0]

    return TFDL2.CropAndResize(Input,AlignMask,outheight,outwidth)

def RoiPool(Input:TFSymbol,Rois:TFSymbol,outheight:int,outwidth:int,spatial_scale:float=1.0)->TFSymbol:
    if type(Input) == tuple or type(Input) == list:
        assert(len(Input)==1)
        Input = Input[0]

    if type(Rois) == tuple or type(Rois) == list:
        assert(len(Rois)==1)
        Rois = Rois[0]

    return TFDL2.RoiPool(Input,Rois,outheight,outwidth,spatial_scale)

def WarpWithOpFlow(Input:TFSymbol,Flow:TFSymbol)->TFSymbol:
    if type(Input) == tuple or type(Input) == list:
        assert(len(Input)==1)
        Input = Input[0]

    if type(Flow) == tuple or type(Flow) == list:
        assert(len(Flow)==1)
        Flow = Flow[0]

    return TFDL2.WarpWithOpFlow(Input,Flow)

def WarpWithAffine(Input:TFSymbol,transMat:TFSymbol,Dsize:tuple=None,padMethod:bool=True)->TFSymbol:
    if type(Input) == tuple or type(Input) == list:
        assert(len(Input)==1)
        Input = Input[0]

    if type(transMat) == tuple or type(transMat) == list:
        assert(len(transMat)==1)
        transMat = transMat[0]

    if Dsize == None:
        Dsize = (0,0)
    else:
        assert(len(Dsize)==2)

    return TFDL2.WarpWithOpAffine(Input,transMat,Dsize[0],Dsize[1],padMethod)

def WarpWithPespect(Input:TFSymbol,transMat:TFSymbol,Dsize:tuple=None,padMethod:bool=True)->TFSymbol:
    if type(Input) == tuple or type(Input) == list:
        assert(len(Input)==1)
        Input = Input[0]

    if type(transMat) == tuple or type(transMat) == list:
        assert(len(transMat)==1)
        transMat = transMat[0]

    if Dsize == None:
        Dsize = (0,0)
    else:
        assert(len(Dsize)==2)

    return TFDL2.WarpWithOpPerspect(Input,transMat,Dsize[0],Dsize[1],padMethod)

def WarpWithGridFace(Input:TFSymbol,transMat:TFSymbol,gridSize:int=16,Dsize:tuple=None,padMethod:bool=True)->TFSymbol:
    if type(Input) == tuple or type(Input) == list:
        assert(len(Input)==1)
        Input = Input[0]

    if type(transMat) == tuple or type(transMat) == list:
        assert(len(transMat)==1)
        transMat = transMat[0]

    if Dsize == None:
        Dsize = (0,0)
    else:
        assert(len(Dsize)==2)

    return TFDL2.WarpWithOpGridFace(Input,transMat,gridSize,Dsize[0],Dsize[1],padMethod)

def Custom(Inputs:tuple,Outputs:tuple,Opname:str,jsonconfig:str)->TFSymbol:
    return TFDL2.Custom(Inputs,Outputs,Opname,jsonconfig)

def Where(Cond:TFSymbol,Input0,Input1)->TFSymbol:
    if type(Input0) == tuple:
        assert(len(Input0)==1)
        Input0 = Input0[0]
    if type(Input1) == tuple:
        assert(len(Input1)==1)
        Input1 = Input1[0]
    if type(Input0) is TFSymbol and type(Input1) is TFSymbol:
        return TFDL2.Where(Cond,Input0,Input1)
    elif type(Input0) is TFSymbol and type(Input1) is not TFSymbol:
        assert(type(Input1) is float or type(Input1) is int)
        if type(Input1) is float:
            return TFDL2.Where1(Cond,Input0,Input1)
        else:
            return TFDL2.Where2(Cond,Input0,Input1)
    elif type(Input0) is not TFSymbol and type(Input1) is TFSymbol:
        assert(type(Input0) is float or type(Input0) is int)
        if type(Input1) is float:
            return TFDL2.Where3(Cond,Input0,Input1)
        else:
            return TFDL2.Where4(Cond,Input0,Input1)
    elif type(Input0) is not TFSymbol and type(Input1) is not TFSymbol:
        assert(type(Input0) is float or type(Input0) is int)
        assert(type(Input1) is float or type(Input1) is int)
        if type(Input1) is float:
            return TFDL2.Where5(Cond,Input0,Input1)
        else:
            return TFDL2.Where6(Cond,Input0,Input1)
    else:
        raise RuntimeError("input type Wrong")

def Crop(Input:TFSymbol,Crop:TFSymbol)->TFSymbol:
    return TFDL2.Crop(Input,Crop)

def Crop3(Input:TFSymbol,crops:tuple,startAxis:int)->TFSymbol:
    for one in crops:
        assert type(one) is tuple
        assert len(one) == 3
    return TFDL2.Crop3(Input,crops,startAxis)

def Range(start:TFSymbol,limit:TFSymbol,step:TFSymbol)->TFSymbol:
    return TFDL2.Range(start,limit,step)

def ReArrange(input:TFSymbol,mode:int)->TFSymbol:
    return TFDL2.ReArrange(input,mode)

def deReArrange(input:TFSymbol,mode:int,oldshape:tuple)->TFSymbol:
    assert type(oldshape) is tuple
    return TFDL2.deReArrange(input,mode,oldshape)

def MaskSoftmax(input:TFSymbol,axis:int,upper:bool,diagonals:int,mod:int)->TFSymbol:
    return TFDL2.MaskSoftmax(input,axis,upper,diagonals,mod)

def Crop2(Input:TFSymbol,start:TFSymbol,end:TFSymbol,axis:TFSymbol=None,step:TFSymbol=None)->TFSymbol:
    assert start is not None
    assert end is not None
    if axis is None:
        axis = TFSymbol()

    if step is None:
        step = TFSymbol()

    return TFDL2.Crop2(Input,start,end,axis,step)

def Cast(Input:TFSymbol,dstType:TFDataType)->TFSymbol:
    return TFDL2.Cast(Input,dstType.value)

def Ceil(Input:TFSymbol)->TFSymbol:
    return TFDL2.Ceil(Input)

def Floor(Input:TFSymbol)->TFSymbol:
    return TFDL2.Floor(Input)

def Pow(Input:TFSymbol,power:int)->TFSymbol:
    return TFDL2.Pow(Input,power)

def Sqrt(Input:TFSymbol)->TFSymbol:
    return TFDL2.Sqrt(Input)

def Log2(Input:TFSymbol)->TFSymbol:
    return TFDL2.Log2(Input)

def Shape(Input:TFSymbol)->TFSymbol:
    return TFDL2.ShapeOp(Input)

def Expand(Input:TFSymbol,shape:TFSymbol)->TFSymbol:
    return TFDL2.Expand(Input,shape)

def ExpandDims(Input:TFSymbol,dims:tuple)->TFSymbol:
    if isinstance(dims,int):
        dims = (dims,)
    elif isinstance(dims,list):
        dims = tuple(dims)
    elif isinstance(dims,tuple):
        pass
    else:
        raise RuntimeError("Wrong kernel type")
    return TFDL2.Expand_dims(Input,dims)

def GreedyCTC(Input:TFSymbol)->TFSymbol:
    return TFDL2.GreedyCTC(Input)

def LayerNorm(Input:TFSymbol,axis:int=-1)->TFSymbol:
    return TFDL2.LayerNorm(Input,axis)

def LayerNorm2(Input:TFSymbol,scale:TFSymbol,bias:TFSymbol,axis:int=-1)->TFSymbol:
    assert scale is not None
    assert bias is not None

    return TFDL2.LayerNorm2(Input,axis,scale,bias)

def ConstantOfShape(Input:TFSymbol,dtype:TFDataType,value:np.ndarray)->TFSymbol:
    assert value.size == 1
    return TFDL2.ConstantOfShape(Input,dtype.value,value)

def Unfold(Input: TFSymbol,kernel,pad,stride,dilation:int,reverse:bool=False)->TFSymbol:
    if isinstance(kernel,int):
        kernel = (kernel,kernel)
    elif isinstance(kernel,list):
        assert len(kernel) == 2
        kernel = tuple(kernel)
    elif isinstance(kernel,tuple):
        assert len(kernel) == 2
        kernel = kernel
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(pad,int):
        pad = (pad,pad,pad,pad)
    elif isinstance(pad,list):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")

    elif isinstance(pad,tuple):
        if len(pad) == 2:
            pad = (pad[0],pad[0],pad[1],pad[1])
        elif len(pad) == 4:
            pad = tuple(pad)
        else:
            raise RuntimeError("Wrong pad size")
        pad = pad
    else:
        raise RuntimeError("Wrong kernel type")

    if isinstance(stride,int):
        stride = (stride,stride)
    elif isinstance(stride,list):
        assert len(stride) == 2
        stride = tuple(stride)
    elif isinstance(stride,tuple):
        assert len(stride) == 2
        stride = stride
    else:
        raise RuntimeError("Wrong kernel type")

    return TFDL2.Unfold(Input,kernel,pad,stride,dilation,reverse)

def ArgMax(Input:TFSymbol,dim:int,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ArgMax(Input,dim,keep_dims)

def ArgMin(Input:TFSymbol,dim:int,keep_dims:bool=True)->TFSymbol:
    if type(Input) == tuple:
        assert(len(Input)==1)
        Input = Input[0]
    return TFDL2.ArgMin(Input,dim,keep_dims)

def Requantize(Input:TFSymbol,maptable:list)->TFSymbol:
    assert(len(maptable) == 256)
    return TFDL2.Requantize(Input,maptable)

def NextFrame(Input:TFSymbol)->TFSymbol:
    '''
    all RNN struct need use this op as last op to package output.
    :param Input:
    :return:
    '''
    return TFDL2.NextFrame(Input)

def LSTM(x:TFSymbol,hidden_size:int,direction:str)->TFSymbol:
    '''
    all RNN struct need use this op as last op to package output.
    :param Input:
    :return:
    '''
    return TFDL2.LSTM(x,hidden_size,direction)

def LSTM2(x:TFSymbol,w:TFSymbol,r:TFSymbol,b:TFSymbol,hidden_size:int,direction:str)->TFSymbol:
    '''
    all RNN struct need use this op as last op to package output.
    :param Input:
    :return:
    '''
    return TFDL2.LSTM2(x,w,r,b,hidden_size,direction)

