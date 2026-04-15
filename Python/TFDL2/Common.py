from .TFDL2 import _CalibrationMode,_TFDataType,_TFConvertType,_DECODER_FLAGS
from enum import Enum


class CalibrationMode(Enum):
    Naive = _CalibrationMode.Naive
    MEAN = _CalibrationMode.MEAN
    KLD = _CalibrationMode.KLD
    COVERAGE = _CalibrationMode.COVERAGE

class DECODER_FLAGS(Enum):
    TFCV_BGR = _DECODER_FLAGS.BGR
    TFCV_RGB = _DECODER_FLAGS.RGB
    TFCV_Gray = _DECODER_FLAGS.Gray

class TFDataType(Enum):
    TFDL_UINT8=_TFDataType.UINT8
    TFDL_FLOAT=_TFDataType.FLOAT
    TFDL_INT32=_TFDataType.INT32
    TFDL_FLOAT64 = _TFDataType.FLOAT64
    TFDL_STRING = _TFDataType.STRING
    TFDL_INT64 = _TFDataType.INT64
    TFDL_FLOAT16 = _TFDataType.FLOAT16
    TFDL_BFLOAT16 = _TFDataType.BFLOAT16
    TFDL_INT16 = _TFDataType.INT16

class TFConverType(Enum):
    CAFFE2TFDL = _TFConvertType.CAFFE2TFDL
    TENSORFLOW2TFDL = _TFConvertType.TENSORFLOW2TFDL
    ONNX2TFDL = _TFConvertType.ONNX2TFDL
    TFLITE2TFDL = _TFConvertType.TFLITE2TFDL