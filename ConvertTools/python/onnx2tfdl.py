from TFConvertor import OnnxConvertor,TFConvertor,OnnxNodeConvertor,NodeConvertor
import argparse
from TFDL2 import TFDataType,Op,DECODER_FLAGS
from collections import OrderedDict
import numpy as np
import glob
import os
from typing import Dict, List, Union

class InputShapeAction(argparse.Action):
    """自定义 Action 来处理输入形状"""
    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        for value in values:
            if ':' not in value:
                parser.error(f"无效格式: {value}。请使用 inputname:1,3,224,224 格式")
            
            name, shape_str = value.split(':', 1)
            name = name.strip()
            
            try:
                shape_values = [int(x.strip()) for x in shape_str.split(',')]
                if not shape_values:
                    parser.error(f"形状值不能为空: {value}")
                result[name] = shape_values
            except ValueError:
                parser.error(f"无效的形状值: {shape_str}")
        
        setattr(namespace, self.dest, result)

def from_onnx(args):
    assert args.onnxpath is not None
    convertor = OnnxConvertor(args.output_name)
    convertor.load(args.onnxpath,stoptensor=args.stoptensor)
    #这里后面可以使用自定义的模型优化器，比如合并算子，剪枝算子等
    convertor.optmize()#([custom1,custom3])#([custom1,custom2])
    #这里可以注册自定义的节点转换器，比如自定义的节点类型，或者需要特殊处理的节点类型，后面函数的书写方法参考customop.py
    #convertor.registorCustomNodeConvertor("PPScatterPlugin",PillarScatterPlugin)
    #convertor.registorCustomNodeConvertor("Optimize1",Optimize1)
    
    convertor.buildTFmodel(inputshape=args.input_shapes,std=args.std,mean=args.mean)
    if args.check:
        if convertor.verification(targetModel=args.testonnxpath):
            convertor.dump("./{0}".format(args.output_name))
            print("Convert Success!!!")
            if args.quantimgDir is not None:
                imglist = glob.glob(os.path.join(args.quantimgDir,"*.jpg"))
                convertor.quantContext(calibration_list=imglist,decoderflags=DECODER_FLAGS(args.cvtype),MergeConcate=False)
            else:
                print("没有量化图片集，使用随机数量化")
                convertor.quantContext(calibration_list=None,decoderflags=DECODER_FLAGS(args.cvtype),MergeConcate=False)
            convertor.dump("./{0}.quant".format(args.output_name))
        else:
            print("Convert Fail!")
    else:
        convertor.dump("./{0}".format(args.output_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='from_onnx.py')
    parser.add_argument('--onnxpath', type=str, help='用来转化的onnx模型地址',required=True)
    parser.add_argument('--testonnxpath', type=str,help='如果onnx模型是修改过的无法直接forward,此时使用这个备用onnx模型地址,如果没有修改onnx则不需要指定此参数')
    parser.add_argument('--output_name', type=str, help='导出模型名称',required=True)
    parser.add_argument('--stoptensor', type=str,nargs='+', help='模型转化停止节点名称',default=None)
    parser.add_argument('--quantimgDir', type=str, help='使用的量化图片集的文件夹地址')
    parser.add_argument('--cvtype', type=int, help='模型接受的输入图片通道类型 0:BGR 1:RGB 2Gray',default=1)
    parser.add_argument('--mean', type=float, nargs='+', help='preproce mean')
    parser.add_argument('--std', type=float, nargs='+', help='preprocess std')
    parser.add_argument('--input_shapes', type=str, nargs='+', action=InputShapeAction,
                   help='如果原始onnx模型输入是dynamic的,为了能顺利比较转化是否成功需要手动指定模型的输入尺寸，输入形状，格式: inputname:1,3,224,224 inputname2:1,3,224,224')
    parser.add_argument('--check', action='store_true', help='是否比较转化的模型与原始onnx的输出结果,从而判断转化是否成功')
    args = parser.parse_args()
    from_onnx(args)

