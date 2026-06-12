import numpy as np
import onnx
from onnx import numpy_helper,external_data_helper,helper
from .TFConvertor import TFConvertor,NodeConvertor,tfdtype2npdtype,npdtype2tfdtype
from TFDL2.Op import *
from TFDL2 import TFConverType,TFSymbol
from TFDL2 import TFExecutor,Option
from .Optimize.MergeBatchnorm import MergeBatchNormOnnx,ReplaceBatchNorm
from .Optimize.MergeHardSwish import MergeHardSwishOnnx
from .Optimize.MergeHardSigmoid import MergeHardSigmoidOnnx
from .Optimize.MergeLayerNorm import MergeLayerNormOnnx
from .Optimize.MergeSwish import MergeSwishOnnx
from .Optimize.PrepareParam import ReshapConvTransposeWeightOnnx,ReshapGemmWeightOnnx,RemoveCastOnnx
from .Optimize.Clip2ReLUX import Clip2RuLUXOnnx
from .Optimize.MergeBiasAdd import MergeBiasAddOnnx
from .Optimize.MergeGeLU import MergeGeLUOnnx
from .Optimize.MergeMaskSoftmax import MergeMaskSoftmaxOnnx
from .Optimize.MergeBert import MergeBertOnnx,MergeConvBertOnnx
from .Optimize.replacedilationconv import ReplaceDilationConv
from .Optimize.SplitConv import SplitConvOnnx
def parseAttr(attr):
    if attr.type == 1:
        return attr.f
    elif attr.type == 2:
        return attr.i
    elif attr.type == 3:
        return str(attr.s,encoding='utf-8')
    elif attr.type == 4:
        return numpy_helper.to_array(attr.t)
    elif attr.type == 5:
        raise NotImplementedError("{} not implemented".format("graph type"))
    elif attr.type == 6:
        return list(attr.floats)
    elif attr.type == 7:
        return list(attr.ints)
    elif attr.type == 8:
        res = []
        for tt in attr.strings:
            res.append(str(tt,encoding='utf-8'))
        return res
    elif attr.type == 9:
        res = []
        for tt in attr.tensors:
            res.append(numpy_helper.to_array(tt))
        return res

def onnxDytpe2TFDtype(dtype):
    if dtype == onnx.TensorProto.FLOAT:
        return TFDataType.TFDL_FLOAT
    elif dtype == onnx.TensorProto.UINT8:
        return TFDataType.TFDL_UINT8
    elif dtype == onnx.TensorProto.INT16:
        return TFDataType.TFDL_INT16
    elif dtype == onnx.TensorProto.INT32:
        return TFDataType.TFDL_INT32
    elif dtype == onnx.TensorProto.INT64:
        return TFDataType.TFDL_INT64
    elif dtype == onnx.TensorProto.FLOAT16:
        return TFDataType.TFDL_FLOAT16
    elif dtype == onnx.TensorProto.DOUBLE:
        return TFDataType.TFDL_FLOAT64
    elif dtype == onnx.TensorProto.STRING:
        return TFDataType.TFDL_STRING
    else:
        raise NotImplementedError("{} not implemented".format(dtype))

class OnnxParser(object):
    @classmethod
    def get_converter(cls, opset):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        versions = [int(d.replace("_impl_v", "")) for d in dir(cls) if "_impl_v" in d]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError("opset version {} of {} not implemented".format(version, cls.__name__))

class ConvolutionParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        kernel_shape = node.attr["kernel_shape"]
        if len(kernel_shape) == 1:
            kernel_shape = kernel_shape[0]
        strides = node.attr["strides"]
        if len(strides) == 1:
            strides = strides[0]
        
        if "pads" not in node.attr and "auto_pad" in node.attr:
            if "auto_pad" in node.attr and node.attr["auto_pad"] == "SAME_UPPER":
                if(kernel_shape[0] == 2 and strides[0] == 1):
                    pads = [0,0,1,1]
                elif(kernel_shape[0] == 3 and strides[0] == 1):
                    pads = [1,1,1,1]
                elif(kernel_shape[0] == 3 and strides[0] == 2):
                    pads = [0,0,1,1]
                elif(kernel_shape[0] == 1):
                    pads = [0,0,0,0]
                else:
                    raise Exception(f"unknown pad {kernel_shape} {strides}")
            else:
                raise Exception("unknown pad")
        else:
            pads = node.attr["pads"]    
        if len(pads) == 1:
            pads = pads[0]
        else:
            pads = [pads[0],pads[2],pads[1],pads[3]]
        group = node.attr["group"]
        assert node.attr["dilations"][0] == node.attr["dilations"][1],"dilations must eq"
        dilations = node.attr["dilations"][0]
        Bias = None
        if len(args) > 2:
            Bias = args[2]

        outchannel = node.graph.params[node.inputs[1]].shape[0]

        return Convolution2(args[0],args[1],Bias,kernel_shape,pads,strides,dilations,outchannel,group)

class PoolingParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if node.op == "GlobalAveragePool":
            return GlobalAvePooling(args[0])
        else:
            kernel_shape = node.attr["kernel_shape"]
            if len(kernel_shape) == 1:
                kernel_shape = kernel_shape[0]
            strides = node.attr["strides"]
            if len(strides) == 1:
                strides = strides[0]
            if "pads" not in node.attr:
                if "auto_pad" in node.attr and node.attr["auto_pad"] == "SAME_UPPER":
                    if(kernel_shape[0] == 2 and strides[0] == 1):
                        pads = [0,0,1,1]
                    else:
                        raise Exception("unknown pad")
                else:
                    raise Exception("unknown pad")
            else:
                pads = node.attr["pads"]
            if len(pads) == 1:
                pads = pads[0]
            else:
                pads = [pads[0],pads[2],pads[1],pads[3]]
            if "ceil_mode" in node.attr.keys():
                ceil_mode = bool(node.attr["ceil_mode"])
            else:
                ceil_mode = False
            if node.op == "AveragePool":
                return AvePooling(args[0],kernel_shape,pads,strides,ceil_mode)
            else:
                return MaxPooling(args[0],kernel_shape,pads,strides,ceil_mode)

class MatMulParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):

        transB = node.attr["transB"] == 1 if "transB" in node.attr else False
        transA = node.attr["transA"] == 1 if "transA" in node.attr else False
        Bias = None
        bias2Rows = False
        if len(args) > 2:
            Bias = args[2]
            if str(args[1]) in node.graph.params:
                channels1 = node.graph.params[str(args[1])].shape[0]
                channels2 = node.graph.params[str(args[2])].shape[0]
                bias2Rows = (channels1 == channels2)
            else:
                channels1 = node.graph.params[str(args[0])].shape[0]
                channels2 = node.graph.params[str(args[2])].shape[0]
                bias2Rows = (channels1 == channels2)
            

        return MatMul2(args[0],args[1],Bias,transA,transB,bias2Rows)

class GemmParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):   
        #transB = node.attr["transB"] == 1 if "transB" in node.attr else False
        #transA = node.attr["transA"] == 1 if "transA" in node.attr else False
        #Bias = None
        #bias2Rows = False
        channels1 = node.graph.params[str(args[1])].shape[0]
        return InnerProduct2(args[0],args[1],None if len(args) < 3 else args[2],channels1)

class ConvTransposeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        kernel_shape = node.attr["kernel_shape"]
        if len(kernel_shape) == 1:
            kernel_shape = kernel_shape[0]
        strides = node.attr["strides"]
        if len(strides) == 1:
            strides = strides[0]
        pads = node.attr["pads"]
        if len(pads) == 1:
            pads = pads[0]
        else:
            pads = [pads[0],pads[2],pads[1],pads[3]]
        group = node.attr["group"]
        assert node.attr["dilations"][0] == node.attr["dilations"][1],"dilations must eq"
        dilations = node.attr["dilations"][0]
        outPadH = 0
        outPadW = 0
        if "output_padding" in node.attr:
            if isinstance(node.attr["output_padding"],list):
                if len(node.attr["output_padding"]) > 2:
                    raise Exception("deconv only support same outPadH or outpadW")
                outPadH = node.attr["output_padding"][0]
                if len(node.attr["output_padding"]) == 2:
                    outPadW = node.attr["output_padding"][1]
                else:
                    outPadW = outPadH
            elif isinstance(node.attr["output_padding"],int):
                outPadW = node.attr["output_padding"]
                outPadH = node.attr["output_padding"]
        Bias = None
        if len(args) > 2:
            Bias = args[2]

        outchannel = node.graph.params[node.inputs[1]].shape[0]

        return DeConvolution2(args[0],args[1],Bias,kernel_shape,pads,strides,dilations,outchannel,group,outPadH,outPadW)

class ActivationParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if node.op == "LeakyRelu":
            return LeakyReLU(args[0],node.attr["alpha"])
        elif node.op == "Relu6":
            return ReLUX(args[0],6.0)
        elif node.op == "ReluX":
            return ReLUX(args[0],node.attr["max"])
        elif node.op == "Tanh":
            return Tanh(args[0])
        elif node.op == "Relu":
            return ReLU(args[0])
        elif node.op == "Exp":
            return Exp(args[0])
        elif node.op == "Sigmoid":
            return Sigmoid(args[0])
        elif node.op == "Swish":
            return Swish(args[0])
        elif node.op == "Mish":
            return Mish(args[0])
        elif node.op == "HardSigmoid":
            if "alpha" in node.attr or "beta" in node.attr:
                return HardSigmoid2(args[0],node.attr['alpha'] if "alpha" in node.attr else 1.0,node.attr['beta'] if "beta" in node.attr else 0.0)
            return HardSigmoid(args[0])
        elif node.op == "HardSwish":
            return HardSwish(args[0])
        elif node.op == "GeLU":
            return GeLU(args[0])
        else:
            raise NotImplementedError("{} not implemented".format(node.op))



class SoftmaxParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Softmax(args[0],node.attr["axis"])

class ResizeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if len(node.inputs) > 3:
            scale = args[2] if str(args[2]).startswith("TFDL_") else (node.graph.params[str(args[2])] if str(args[2]) in node.graph.params else None)
            size = args[3] if str(args[3]).startswith("TFDL_") else (node.graph.params[str(args[3])] if str(args[3]) in node.graph.params else None)
        else:
            scale = args[2] if str(args[2]).startswith("TFDL_") else (node.graph.params[str(args[2])] if str(args[2]) in node.graph.params else None)
            size = None

        if node.attr["mode"] == "nearest":
            if scale is not None and type(scale) == str:
                raise Exception("不支持 scale 作为输入，必须作为参数")
            elif scale is not None and type(scale) == np.ndarray and scale.size > 0:
                assert scale.size == 4 and scale[2] == scale[3]
                return NearestReSize2(args[0],scale=scale[3])
            elif size is not None and type(size) == TFSymbol:
                return NearestReSize3(args[0],size)
            elif size is not None and type(size) == np.ndarray and size.size > 0:
                assert size.size == 4 
                return NearestReSize1(args[0],size[2],size[3])
            else:
                raise Exception("Unknown Resize type")
        elif node.attr["mode"] == "linear":
            assert 'coordinate_transformation_mode' in node.attr.keys(), "coordinate_transformation_mode must in attr"
            align_corners = node.attr['coordinate_transformation_mode'] == 'align_corners'
            half_pixel = node.attr['coordinate_transformation_mode'] == 'half_pixel'
            assert align_corners or half_pixel, "coordinate_transformation_mode must be align_corners or half_pixel"

            if scale is not None and type(scale) == str:
                raise Exception("不支持 scale 作为输入，必须作为参数")
            elif scale is not None and type(scale) == np.ndarray and scale.size > 0:
                assert scale.size == 4 and scale[2] == scale[3]
                return BilnearReSize2(args[0],scale=scale[3],align_corners=align_corners)
            elif size is not None and type(size) == TFSymbol:
                return BilnearReSize3(args[0],size,align_corners)
            elif size is not None and type(size) == np.ndarray and size.size > 0:
                assert size.size == 4 
                return BilnearReSize1(args[0],size[2],size[3],align_corners)
            else:
                raise Exception(f"Unknown Resize type {scale} {size}")
        else:
            raise NotImplementedError("{} not implemented".format(node.attr["mode"]))

class ShapeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Shape(args[0])

class ReshapeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if len(args) > 1:
            return Reshape2(*args)
        else:
            return Reshape(args[0],node.attr["shape"])

class GatherParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
       return Gather(args[0],args[1],node.attr["axis"] if "axis" in node.attr else 0)

class BinaryOpParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        assert len(args) == 2,"BinaryOp need two inputs"
        if node.op == "Add":
            return args[0] + args[1]
        elif node.op == "Mul":
            return args[0] * args[1]
        elif node.op == "Sub":
            return args[0] - args[1]
        elif node.op == "Div":
            return args[0] / args[1]

class ConcatOpParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Concat(tuple(args),node.attr["axis"])

class SliceOpParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if "starts" in node.attr and "ends" in node.attr and 'axes' in node.attr:
            return Crop3(args[0],startAxis=node.attr['axes'][0],crops=((node.attr["starts"][0],node.attr['ends'][0]-1,1),))
        return Crop2(*args)

class PadOpParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Pad2(*args)

class CastParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Cast(args[0],onnxDytpe2TFDtype(node.attr["to"]))

class UnsqueezeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if "axes" not in node.attr:
            if len(args) == 1:
                raise Exception("Unsqueeze need at least two inputs")
            if str(args[1]) not in node.graph.params:
                raise Exception("Unsqueeze need axes as param not input")
            
            return ExpandDims(args[0],tuple(node.graph.params[str(args[1])].tolist()))
        else:
            return ExpandDims(args[0],node.attr["axes"])

class SqueezeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if "axes" not in node.attr and len(args) == 1:
            return Squeeze(args[0],(0,))
        else:
            if "axes" not in node.attr:
                return Squeeze(args[0],tuple(node.graph.params[node.inputs[1]].tolist()))     
            return Squeeze(args[0],tuple(node.attr["axes"]))

class SplitParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if "split" not in node.attr:
            return Slice(args[0],node.attr["axis"],tuple(node.graph.params[node.inputs[1]].tolist()))
        return Slice(args[0],node.attr["axis"],tuple(node.attr["split"]))

class CropParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Crop3(args[0],node.attr["crops"],node.attr["startAxis"])

class ReduceMeanParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return ReduceMean(args[0],tuple(node.attr["axes"]),bool(node.attr["keepdims"]))

class ReduceMaxParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return ReduceMax(args[0],tuple(node.attr["axes"]),bool(node.attr["keepdims"]))

class ReduceMinParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return ReduceMin(args[0],tuple(node.attr["axes"]),bool(node.attr["keepdims"]))
    


class MathOpParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if node.op == "Floor":
            return Floor(args[0])
        elif node.op == "Equal":
            assert len(args) == 2
            return EQ(args[0],args[1])
        else:
            raise NotImplementedError("{} not implemented".format(node.op))

class LayerNormalParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if len(args) == 3:
            return LayerNorm2(*args)
        else:
            return LayerNorm2(args[0],args[-2],args[-1])

class ConstantOfShapeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        value = node.attr["value"]
        dtype = npdtype2tfdtype(value.dtype.type)
        print(value,dtype)
        return ConstantOfShape(args[0],dtype,value=value)

class WhereParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Where(*args)

class TransposeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Transpose(args[0],tuple(node.attr["perm"]))

class RangeParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Range(*args)

class ArgParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        if node.op == "ArgMax":
            return ArgMax(args[0],node.attr["axis"],bool(node.attr["keepdims"]) if "keepdims" in node.attr else True)
        elif node.op == "ArgMin":
            return ArgMin(args[0],node.attr["axis"],bool(node.attr["keepdims"]) if "keepdims" in node.attr else True)

class FlattenParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Flatten2Matrix(args[0],node.attr["axis"])

class MaskSoftmaxParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return MaskSoftmax(args[0],node.attr["axis"],node.attr["upper"],node.attr["diagonals"],node.attr["mod"])

class ScaleParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        return Scale2(args[0],args[1],args[2] if len(args)>2 else None)

class TFUnfoldParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        a = Squeeze(args[0],(0,))
        a = ExpandDims(a,(1,))
        a = Unfold(a,kernel=node.attr["kernel_size"],pad=1,stride=1,dilation=1,reverse=True)
        a = Reshape(a,(1,-1,node.attr["kernel_size"][0],node.attr["kernel_size"][1]))
        return a

class ExpandParser(OnnxParser):
    @classmethod
    def _impl_v11(cls,*args,node:NodeConvertor):
        assert len(args) == 2
        return Expand(args[0],args[1])

def _get_convert_map(opset):
    return {
        "Conv":ConvolutionParser.get_converter(opset),
        "GlobalAveragePool":PoolingParser.get_converter(opset),
        "MaxPool":PoolingParser.get_converter(opset),
        "AveragePool":PoolingParser.get_converter(opset),
        "MatMul":MatMulParser.get_converter(opset),
        "Gemm":GemmParser.get_converter(opset),
        "ConvTranspose":ConvTransposeParser.get_converter(opset),
        "Relu":ActivationParser.get_converter(opset),
        "Relu6":ActivationParser.get_converter(opset),
        "ReluX":ActivationParser.get_converter(opset),
        "Swish":ActivationParser.get_converter(opset),
        "HardSigmoid":ActivationParser.get_converter(opset),
        "HardSwish":ActivationParser.get_converter(opset),
        "Tanh":ActivationParser.get_converter(opset),
        "Elu":ActivationParser.get_converter(opset),
        "Sigmoid":ActivationParser.get_converter(opset),
        "Exp":ActivationParser.get_converter(opset),
        "LeakyRelu":ActivationParser.get_converter(opset),
        "GeLU":ActivationParser.get_converter(opset),
        "Softmax":SoftmaxParser.get_converter(opset),
        "Upsample":ResizeParser.get_converter(opset),
        "Resize":ResizeParser.get_converter(opset),
        "Shape":ShapeParser.get_converter(opset),
        "Reshape":ReshapeParser.get_converter(opset),
        "Gather":GatherParser.get_converter(opset),
        "Add":BinaryOpParser.get_converter(opset),
        "Mul":BinaryOpParser.get_converter(opset),
        "Sub":BinaryOpParser.get_converter(opset),
        "Div":BinaryOpParser.get_converter(opset),
        "Concat":ConcatOpParser.get_converter(opset),
        "Slice":SliceOpParser.get_converter(opset),
        "Pad":PadOpParser.get_converter(opset),
        "Cast":CastParser.get_converter(opset),
        "Unsqueeze":UnsqueezeParser.get_converter(opset),
        "Floor":MathOpParser.get_converter(opset),
        "Equal":MathOpParser.get_converter(opset),
        "Split":SplitParser.get_converter(opset),
        "InstanceNormalization":LayerNormalParser.get_converter(opset),
        "LayerNorm":LayerNormalParser.get_converter(opset),
        "LayerNormalization":LayerNormalParser.get_converter(opset),
        "ConstantOfShape":ConstantOfShapeParser.get_converter(opset),
        "Where":WhereParser.get_converter(opset),
        "Transpose":TransposeParser.get_converter(opset),
        "Range":RangeParser.get_converter(opset),
        "ArgMax":ArgParser.get_converter(opset),
        "ArgMin":ArgParser.get_converter(opset),
        "Flatten":FlattenParser.get_converter(opset),
        "MaskSoftmax":MaskSoftmaxParser.get_converter(opset),
        "Crop":CropParser.get_converter(opset),
        "Squeeze":SqueezeParser.get_converter(opset),
        "ReduceMean":ReduceMeanParser.get_converter(opset),
        "TFUnfold":TFUnfoldParser.get_converter(opset),
        "Scale":ScaleParser.get_converter(opset),
        "Expand":ExpandParser.get_converter(opset),
        "ReduceMax":ReduceMaxParser.get_converter(opset),
        "ReduceMin":ReduceMinParser.get_converter(opset),
    }

class OnnxNodeConvertor(NodeConvertor):
    def __init__(self,graph):
        super(OnnxNodeConvertor,self).__init__(graph=graph)

    def load(self,proto:object):
        self.proto = proto
        self.inputs = list(proto.input)
        self.name = proto.name
        self.outputs = list(proto.output)
        if self.name == "":
            self.name = self.outputs[0]
        self.op = proto.op_type
        for attr in proto.attribute:
            self.attr[attr.name] = parseAttr(attr=attr)

    def __call__(self, *args, **kwargs):
        opsetConvertors = _get_convert_map(self.graph.opset)
        if self.op in self.graph.customNodeConvertormap:
            return self.graph.customNodeConvertormap[self.op](node=self,*args)
        else:
            if self.op in opsetConvertors:
                return opsetConvertors[self.op](node=self,*args)
            else:
                raise NotImplementedError("{} not implemented".format(self.op))


class OnnxConvertor(TFConvertor):
    def __init__(self,name:str):
        self.type = TFConverType.ONNX2TFDL
        super(OnnxConvertor,self).__init__(name)

    def load(self,path:str,stoptensor=None):
        assert path is not None
        self.onnxmodel = onnx.load_model(path)
        if type(stoptensor) is not list:
            self.stoptensor = [stoptensor]
        else:
            self.stoptensor = stoptensor
        self.opset = self.onnxmodel.opset_import[0].version
        self.producer_name = self.onnxmodel.producer_name
        self.producer_version = self.onnxmodel.producer_version
        #load all weight
        for pp in self.onnxmodel.graph.initializer:
            self.params[pp.name] = numpy_helper.to_array(pp)

        ignoreTensors = {}
        for stopname in self.stoptensor:
            ignoreTensors[stopname] = set([stopname])
        # load all node and constant
        for nodeproto in self.onnxmodel.graph.node:
            nodeconvert = OnnxNodeConvertor(self)
            nodeconvert.load(nodeproto)
            if nodeconvert.op == "Constant":
                '''
                if hasattr(nodeconvert.attr["value"],"size"):
                    if nodeconvert.attr["value"].size == 1:
                        self.params[nodeconvert.name] = nodeconvert.attr["value"].flatten()[0]
                    else:
                        continue
                else:
                    self.params[nodeconvert.name] = nodeconvert.attr["value"]
                '''
                if type(nodeconvert.attr['value']) is np.ndarray:
                    self.params[nodeconvert.name] = nodeconvert.attr["value"]
                else:
                    raise RuntimeError("Constant value is not np")
                continue
            elif nodeconvert.op == "Identity":
                self.params[nodeconvert.name] = self.params[nodeconvert.inputs[0]]
                continue
            isIgnore = False
            for stopname in self.stoptensor:
                if any([inputtensor in ignoreTensors[stopname] for inputtensor in nodeconvert.inputs]):
                    ignoreTensors[stopname].update(nodeconvert.outputs)
                    isIgnore = True
                    break
            if isIgnore:
                continue
            self.nodes[nodeconvert.name] = nodeconvert

        # load model inputs
        for input in self.onnxmodel.graph.input:
            input_name = input.name
            if input_name in self.params:
                continue
            input_type = onnxDytpe2TFDtype(input.type.tensor_type.elem_type)
            input_shape = []
            for dim in input.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value if dim.dim_value > 0 else 1)
            self.inputs[input_name] = {"dtype":input_type,"shape":input_shape}
        
        for output in self.onnxmodel.graph.output:
            for stopname in self.stoptensor:
                if output.name in ignoreTensors[stopname]:
                    self.outputs.append(stopname)    
                    break
            else:
                self.outputs.append(output.name)
                
        self.outputs = list(set(self.outputs))
        removeParams = []
        for name in self.params.keys():
            ifremove = True
            for node in self.nodes.values():
                if name in node.inputs:
                    ifremove = False
                    break
            if ifremove:
                removeParams.append(name)

        for name in removeParams:
            del self.params[name]

    def optmize(self,CustomOptimize:list=[]):
        Clip2RuLUXOnnx(self)
        #ReplaceDilationConv(self)
        ReshapConvTransposeWeightOnnx(self)
        ReshapGemmWeightOnnx(self)
        MergeBiasAddOnnx(self)
        MergeBatchNormOnnx(self)
        MergeHardSwishOnnx(self)
        MergeGeLUOnnx(self)
        MergeHardSigmoidOnnx(self)
        MergeLayerNormOnnx(self)
        MergeSwishOnnx(self)
        MergeMaskSoftmaxOnnx(self)
        ReplaceBatchNorm(self)
        #RemoveCastOnnx(self)
        #RemoveCastOnnx(self)
        #MergeConvBertOnnx(self)
        MergeBertOnnx(self)
        SplitConvOnnx(self)
        for opt in CustomOptimize:
            opt(self)

    def verification(self,targetModel=None,checktensors=[]):
        import onnxruntime
        #newoutput = ["onnx::Cast_163"]
        #self.onnxmodel.graph.output.pop()
        #self.onnxmodel.graph.output.extend([onnx.ValueInfoProto(name=newoutput[0])])

        inputs = {}
        outputs = []
        newoption = Option
        newoption["UseHardware"] = False
        newoption["FrugalMode"] = False
        self.executor = TFExecutor(self._context,newoption)
        tfinputs = self.executor.GetInputs()
        for tfin in tfinputs:
            for key,value in self.renames.items():
                if tfin.name == str(value):
                    tfin.fromNumpy(np.random.randint(0,255,tfin.shape).astype(tfdtype2npdtype(tfin.dtype)))
                    break
        tfout = self.executor()
        for tfin in tfinputs:
            for key,value in self.renames.items():
                if tfin.name == str(value):
                    inputs[key] = tfin.toNumpy()
                    break
        if targetModel is None:
            onnex_model = onnxruntime.InferenceSession(self.onnxmodel.SerializeToString())
            try:
                for oname in self.onnxmodel.graph.output:
                    outputs.append(oname.name)

                onnxout = onnex_model.run(outputs,inputs)
            except Exception as e:
                print("OnnxRunTime Fail, or you can dump tfdl model without verification ")
                print(str(e))
                return False
        else:
            targetModel = onnx.load_model(targetModel)
            # 如果要删除特定输出
            while targetModel.graph.output:
                targetModel.graph.output.pop()

            setoutpunames = [onnx.ValueInfoProto(name=tensorname) for tensorname in checktensors]
            setoutpunames.extend([onnx.ValueInfoProto(name=tensorname) for tensorname in self.outputs])
            targetModel.graph.output.MergeFrom(setoutpunames)
            #targetModel.graph.output.extend(setoutpunames)
            

            onnex_model = onnxruntime.InferenceSession(targetModel.SerializeToString())
            try:
                for oname in targetModel.graph.output:#self.onnxmodel.graph.output:
                    outputs.append(oname.name)

                onnxout = onnex_model.run(outputs,inputs)
            except Exception as e:
                print("OnnxRunTime Fail, or you can dump tfdl model without verification ")
                print(str(e))
                return False

        res = True
        for i,name in enumerate(outputs):
            tfname = str(self.renames[name])
            onnxdata = onnxout[i]
            tfdata = None
            #for tfd in tfout:
            #    if tfd.name == tfname:
            #        tfdata = tfd.toNumpy()
            #        break
            tfdata = self.executor.GetTensorByName(tfname).toNumpy()
            if tfdata is not None:
                onnxdata = onnxdata.flatten()
                tfdata = tfdata.flatten()
                result = (np.dot(tfdata,onnxdata)/(np.linalg.norm(tfdata)*np.linalg.norm(onnxdata)))
                print('{0} l2 norm loss is {1}'.format(name,result))
                print('please check data: tfdl and onnx')
                print(tfdata[:10])
                print(onnxdata[:10])
                if result < 0.9:
                    res = False
                    #break
            else:
                print("{0} {1} can't find in TFDL model".format(name,tfname))
                res = False

        return res





