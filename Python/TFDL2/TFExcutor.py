from .TFDL2 import _TFExecutor,TFSymbol
from .TFTensor import TFTensor
import json

Option = {
    "UseHardware":True,
    "FrugalMode":True,
    "optimize":{
        "DoAlign":False,
        "TryReverse":False,
    },
    "InputShape":[
        #{"NodeName":"","Shape":[]},
        #{}
    ]

}
Modify = {
    "AddOnPass": [],
    "DeleteLayer": [],
    "Layer": [
        #{
        #    "layerName": "TFDL_CONVOLUTION_58",
        #    "outputDataType":  "TFDtypeFp32"
        #},
        #{
        #    "layerName": "TFDL_CONVOLUTION_66",
        #    "outputDataType":  "TFDtypeFp32"
        #},
        #{
        #    "layerName": "TFDL_CONVOLUTION_74",
        #    "outputDataType":  "TFDtypeFp32"
        #}
    ]
}



class TFExecutor(_TFExecutor):
    def __init__(self,context,config:dict):
        super(TFExecutor,self).__init__(context,json.dumps(config))

    def GetInputs(self):
        inputs =  self._Getinputs()
        newinputs = []
        for data in inputs:
            newinputs.append(TFTensor(data))
        return newinputs

    def GetTensorByName(self,Name:str):
        return TFTensor(self._GetTensorByName(Name))

    def SetPrintInfo(self,print:bool):
        self._SetPrintInfo(print)

    def DumpSim(self,path:str):
        self._DumpSim(path)

    def __call__(self, Alone:bool=True):
        outputs =  self._Forward(Alone)
        newoutputs = []
        for data in outputs:
            newoutputs.append(TFTensor(data))
        return newoutputs