from . import TFDL2
from typing import Callable
from .Common import *

def LoadCustomOp(path:str):
    TFDL2.LoadCustomOp(path)

def CustomReshape(*args,**kwargs):
    pass
def CustomEval(*args,**kwargs):
    pass

def RegisterCustomOp(OpName:str,Reshape:Callable,Eval:Callable):
    TFDL2.RegisterCustomOp(OpName,Reshape,Eval)
