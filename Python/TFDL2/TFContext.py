from .TFDL2 import _TFContext,TFSymbol
from .TFTensor import TFTensor
import json
import base64
from typing import Callable
class TFContext(_TFContext):
    def __init__(self,ContextName="",path=None):
        if path:
            super(TFContext,self).__init__((path,))
        else:
            super(TFContext,self).__init__(ContextName)

    def __enter__(self):
        assert(super(TFContext,self)._open())
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        super(TFContext,self)._Close()

    def Modify(self,Config:dict):
        super(TFContext,self)._Modify(json.dumps(Config))

    def Assign(self,tfrom:TFSymbol,to:TFSymbol):
        super(TFContext,self)._Assign(str(tfrom),str(to))

    def Dump(self,path:str):
        super(TFContext,self)._DumpToFile(path)

    def RegisterParamToContext(self,**kwargs):
        super(TFContext,self)._RegisterParamToContext(kwargs)

    def GetParam(self,paramName:str):
        return TFTensor(super(TFContext,self)._GetParam(paramName))

    def GetAttr(self,paramName:str):
        ddd = super(TFContext,self)._GetAttr(paramName)
        ddd = bytes(ddd,encoding='utf-8')
        ddd = base64.decodebytes(ddd)
        return ddd
        #return json.loads(super(TFContext,self)._GetAttr(paramName),)

    def GetParamSymbol(self,paramName:str):
        return super(TFContext,self)._GetParamSymbol(paramName)

    def GetOutSymbols(self)->list:
        return super(TFContext,self)._GetOutSymbols()
    
    def GetInputSymbols(self)->list:
        return super(TFContext,self)._GetInputSymbols()

    def SetOutputs(self,paramName:list):
        for name in paramName:
            assert(type(name) is str)
        return super(TFContext,self)._SetOut(paramName)

    def ReplaceOp(self,NodeName:str,builder:Callable):
        super(TFContext,self)._ReplaceOp(NodeName,builder)

    def SubGraph(self,inputs:tuple,builder:Callable)->TFSymbol:
        super(TFContext,self)._SubGraph(inputs,builder)

    def RemoveOp(self,NodeName:str):
        super(TFContext,self)._RemoveOp(NodeName)

    def AddInt8Config(self,name:str,max:float,min:float)->bool:
        return super(TFContext,self)._RegistorInt8config(name,max,min)