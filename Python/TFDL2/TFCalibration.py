from .TFDL2 import _TFCalibration,TFSymbol
from .TFTensor import TFTensor
import json
from .Common import TFDataType,TFConverType


class TFCalibration(_TFCalibration):
    def __init__(self,context,calibrationmode,config:dict=None):
        if config == None:
            config = {}
            config["UseHardware"] = False
            config["FrugalMode"] = False
            config["optimize"] ={}
            config["InputShape"] =[]
        _TFCalibration.__init__(self,context,calibrationmode.value,json.dumps(config))

    def __call__(self, *args, **kwargs):
        self._Calibration()

    def Quantize(self,inputtype:dict,avoidtensors:tuple=(),stopquanttensors:tuple=(),MergeEltwise:bool=False,MergeConcate:bool=True,Perchannel:bool=True):
        for k in inputtype.keys():
            if isinstance(inputtype[k],TFDataType):
                inputtype[k] = inputtype[k].value
        self._Quantize(inputtype,avoidtensors,stopquanttensors,MergeEltwise,MergeConcate,Perchannel)

    def QuantizeLite(self,inputtype:dict,avoidtensors:tuple=(),stopquanttensors:tuple=(),MergeEltwise:bool=False,MergeConcate:bool=True,Perchannel:bool=True):
        """Quantize weights/ranges without running graph optimization."""
        normalized = {
            name: dtype.value if isinstance(dtype, TFDataType) else dtype
            for name, dtype in inputtype.items()
        }
        self._QuantizeLite(normalized,avoidtensors,stopquanttensors,MergeEltwise,MergeConcate,Perchannel)

    def GetInputs(self):
        inputs =  self._Getinputs()
        newinputs = []
        for data in inputs:
            newinputs.append(TFTensor(data))
        return newinputs
    
    def ConvertCalibrationFp32ToFp16(self):
        self._Fp32ToFp16()
