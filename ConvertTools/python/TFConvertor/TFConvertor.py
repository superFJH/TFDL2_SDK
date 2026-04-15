from TFDL2 import TFContext,TFTensor,TFSymbol,TFDataType,TFConverType,CalibrationMode,DECODER_FLAGS,TFCalibration
from TFDL2.Op import Placeholder2,Custom
import numpy as np
import os
import cv2
from collections import OrderedDict

def npdtype2tfdtype(dtype:np.dtype):
    if dtype is np.float32:
        return TFDataType.TFDL_FLOAT
    elif dtype is np.int32:
        return TFDataType.TFDL_INT32
    elif dtype is np.uint8:
        return TFDataType.TFDL_UINT8
    elif dtype is np.float64:
        return TFDataType.TFDL_FLOAT64
    elif dtype is np.int64:
        return TFDataType.TFDL_INT64
    elif dtype is np.int16:
        return TFDataType.TFDL_INT16
    elif dtype is np.float16:
        return TFDataType.TFDL_FLOAT16

def tfdtype2npdtype(dtype:TFDataType):
    if dtype is TFDataType.TFDL_FLOAT:
        return np.float32
    elif dtype is TFDataType.TFDL_INT32:
        return np.int32
    elif dtype is TFDataType.TFDL_INT16:
        return np.int16
    elif dtype is TFDataType.TFDL_INT64:
        return np.int64
    elif dtype is TFDataType.TFDL_FLOAT64:
        return np.float64
    elif dtype is TFDataType.TFDL_UINT8:
        return np.uint8
    elif dtype is TFDataType.TFDL_FLOAT16:
        return np.float16


class NodeConvertor(object):
    def __init__(self,**kwargs):
        self.inputs = []
        self.outputs = []
        self.attr = {}
        self.op = ""
        self.name = ""
        self.graph = kwargs["graph"]
        super(NodeConvertor,self).__init__()

    def load(self,proto:object):
        pass

    def __str__(self):
        return "{} {}".format(self.name,self.op)

    def __call__(self, *args, **kwargs)->TFSymbol:
        pass

def preprocess(img,inputsize,dtype:np.dtype):
    img = cv2.resize(img,inputsize)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis,:]
    img = img.astype(dtype)
    img = np.ascontiguousarray(img)
    return img

import json
class TFConvertor(object):
    """A helper class for handling Relay expression

        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph
    """
    def __init__(self,name:str):
        self.name = name
        self.nodes = OrderedDict()
        self.params = {}
        # this dict map the model's real name with TFSymbol
        self.renames = {}
        self.nodeMap = {}
        self.num_input = 0
        self.num_param = 0
        self.inputs = {}
        self.outputs = []
        self.customNodeConvertormap = {}
        self.type = TFConverType.CAFFE2TFDL
        super(TFConvertor, self).__init__()

    def load(self,path:str,stoptensor=None):
        pass

    def optmize(self):
        pass

    def buildTFmodel(self,inputshape=None,std=None,mean=None):
        with TFContext(self.name) as context:
            
            for name,attr in self.inputs.items():
                if inputshape is not None:
                    shape = tuple(inputshape[name])
                else:
                    shape = tuple(attr["shape"])
                if std is not None and mean is not None:
                    self.renames[name] = Placeholder2(context,shape,attr["dtype"],scale=tuple(1.0/np.array(std)),mean=tuple(mean))
                else:
                    iiinode = Placeholder2(context,shape,attr["dtype"])
                    self.renames[name] = iiinode

            for name,param in self.params.items():
                assert param is not np.ndarray, "only numpy array can register in TFDL"
            tfoutputname = {}
            TFSymbolout = []
            context.RegisterParamToContext(**self.params)
            nofoundInputnodes = []
            for name,node in self.nodes.items():
                tmpinput = []
                for iname in node.inputs:
                    if iname == "" or iname == "ISNULL":
                        tmpinput.append(None)
                        continue
                    if iname in self.renames:
                        tmpinput.append(self.renames[iname])
                        continue
                    if iname in self.params:
                        tmpinput.append(context.GetParamSymbol(iname))
                        continue
                if len(tmpinput) != len(node.inputs):
                    print("{} not found input backup!!".format(node.name))
                    nofoundInputnodes.append(node)
                    continue
                outputs = node(*tmpinput)

                if isinstance(outputs,list):
                    for index,symbol in enumerate(outputs):
                        self.renames[node.outputs[index]] = symbol
                else:
                    self.nodeMap[node.name] = str(outputs)
                    self.renames[node.outputs[0]] = outputs
                
                for oname in node.outputs:                        
                    if oname in self.outputs:
                        if isinstance(outputs,list):
                            for index,symbol in enumerate(outputs):
                                tfoutputname[oname] = str(symbol)
                                #tfoutputname.append(str(symbol))
                        else:
                            #tfoutputname.append(str(outputs))
                            tfoutputname[oname] = str(outputs)

                print("Convert {} {} --> {}".format(node.name,node.op,str(outputs) if isinstance(outputs,TFSymbol) else [str(o) for o in outputs]))
                #if node.name == "x.3":
                #    break
            while len(nofoundInputnodes) != 0:
                backup = []
                for node in nofoundInputnodes:
                    tmpinput = []
                    for iname in node.inputs:
                        if iname == "" or iname == "ISNULL":
                            tmpinput.append(None)
                            continue
                        if iname in self.renames:
                            tmpinput.append(self.renames[iname])
                            continue
                        if iname in self.params:
                            tmpinput.append(context.GetParamSymbol(iname))
                            continue
                    if(len(tmpinput) != len(node.inputs)):
                        backup.append(node)
                        continue
                    outputs = node(*tmpinput)

                    if isinstance(outputs,list):
                        for index,symbol in enumerate(outputs):
                            self.renames[node.outputs[index]] = symbol
                    else:
                        self.nodeMap[node.name] = str(outputs)
                        self.renames[node.outputs[0]] = outputs

                    for oname in node.outputs:                        
                        if oname in self.outputs:
                            if isinstance(outputs,list):
                                for index,symbol in enumerate(outputs):
                                    #tfoutputname.append(str(symbol))
                                    tfoutputname[oname] = str(symbol)
                                    TFSymbolout.append(symbol)
                            else:
                                #tfoutputname.append(str(outputs))
                                tfoutputname[oname] = str(outputs)
                                TFSymbolout.append(outputs)
                                
                            
                                
                    print("Convert {} {} --> {}".format(node.name,node.op,str(outputs) if isinstance(outputs,TFSymbol) else [str(o) for o in outputs]))
                assert len(backup) != len(nofoundInputnodes), "backup len must not equal nofoundInputnodes len"
                nofoundInputnodes = backup
            #TFSymbolout.append(iiinode)
            #tfoutputname["input"] = str(iiinode)
            #outnode = Custom(tuple(TFSymbolout),("dets","labels"),"fastrcnn_backbone",json.dumps(tfoutputname))

            
        
        print("output : ",list(tfoutputname.values()))
        context.SetOutputs(list(tfoutputname.values()))

        print("Finished!!")

        self._context = context

    def dump(self,path:str):
        self._context.Dump(path)

    def verification(self,targetModel=None,checktensors=[]):
        pass
    #注册自定义算子转化函数，对于onnx中一些不支持的算子，使用这个自定义函数来转化，避免直接报错退出
    def registorCustomNodeConvertor(self,opName,func):
        self.customNodeConvertormap[opName] = func

    def quantContext(self,calibration_list:list=None,quant_input:dict=None,decoderflags:DECODER_FLAGS=None,MergeConcate:bool=True, avoidtensors:tuple=(),stopquanttensors:tuple=()):
        calibration = TFCalibration(self._context,CalibrationMode.Naive)
        inputs = calibration.GetInputs()

        if calibration_list is None:
            for dd in inputs:
                if dd.dtype == TFDataType.TFDL_INT32 or dd.dtype == TFDataType.TFDL_INT64:
                    example = np.random.randint(0,13740,dd.shape).astype(tfdtype2npdtype(dd.dtype))#np.random.randint(0,255,dd.shape).astype(np.float32)
                else:
                    example = np.random.random(dd.shape).astype(tfdtype2npdtype(dd.dtype))#np.random.randint(0,255,dd.shape).astype(np.float32)
                dd.fromNumpy(example)
            calibration()
        else:
            assert len(inputs) == 1, "inputs len must be 1,when run with calibration list"
            for item in calibration_list:
                print(item)
                if os.path.exists(item):
                    if decoderflags == DECODER_FLAGS.TFCV_BGR:
                        img = cv2.imread(item,cv2.IMREAD_COLOR)
                    elif decoderflags == DECODER_FLAGS.TFCV_RGB:
                        img = cv2.imread(item,cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    else:
                        img = cv2.imread(item,cv2.IMREAD_GRAYSCALE)
                    if inputs[0].dtype == TFDataType.TFDL_FLOAT:
                        tensor = preprocess(img,inputsize=(inputs[0].shape[3],inputs[0].shape[2]),dtype=np.float32)
                    else:
                        tensor = preprocess(img,inputsize=(inputs[0].shape[3],inputs[0].shape[2]),dtype=np.uint8)
                    inputs[0].fromNumpy(tensor)
                else:
                    raise Exception(f"can't open {item}")
                calibration()

        stopquanttensors = tuple([str(self.nodeMap[name]) for name in stopquanttensors if name in self.nodeMap])
        avoidtensors = tuple([str(self.nodeMap[name]) for name in avoidtensors if name in self.nodeMap])
        if quant_input is None:
            inputtype ={}
            for data in inputs:
                inputtype[data.name] = TFDataType.TFDL_UINT8    
            calibration.Quantize(inputtype,MergeConcate=MergeConcate,stopquanttensors=stopquanttensors,avoidtensors=avoidtensors)
        else:
            calibration.Quantize(quant_input,MergeConcate=MergeConcate,stopquanttensors=stopquanttensors,avoidtensors=avoidtensors)

