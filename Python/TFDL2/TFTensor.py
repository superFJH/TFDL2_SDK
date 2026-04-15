from .TFDL2 import _TFTensor
from .Common import TFDataType
import numpy as np
class TFTensor(_TFTensor):

    def __init__(self,data:_TFTensor):
        super(TFTensor,self).__init__(data)

    @property
    def name(self):
        return self._name()

    @property
    def shape(self):
        return self._shape()

    @property
    def dtype(self):
        return TFDataType(self._dtype())

    @property
    def qmax(self):
        return self._maxs()

    @property
    def qmin(self):
        return self._mins()

    @property
    def qscale(self):
        return self._scales()

    @property
    def qzeropoint(self):
        return self._zeropoints()

    def toNumpy(self):
        return self._tonumpy()

    def fromNumpy(self,data):
        return self._fromNumpy(np.ascontiguousarray(data))

    def ReadImg(self,path:str):
        return self._ReadImg(path)

    def __str__(self):
        return self.__str__()

    def SetScalar(self,data,dtype=None):
        super(TFTensor,self)._SetScalar(data,dtype)
