#!/bin/bash

export LD_LIBRARY_PATH=$(pwd)/../../lib:$(pwd)/../../lib/CV_NPU10T
if [ -f "../coco.list" ]; then
    echo "量化图片集文件存在"
else
    echo "量化图片集文件不存在"
    exit 1
fi
echo "开始模型转化...."
../../ConvertTools/exe/TFConvertor --proto 4 --quantize 0 --config runconfig.json --input_file ./yolov5su.onnx --output ./yolov5su --scale 0.00392157 0.00392157 0.00392157 --mean 0 0 0 --cv_flags 1
echo "开始模型量化...."
../../ConvertTools/exe/TFConvertor --proto 5 --quantize 1 --calibration_mode 0 --calibration_list ../coco.list --config runconfig.json --input_file ./yolov5su.fb --output yolov5su.quant --scale 0.00392157 0.00392157 0.00392157 --mean 0 0 0 --cv_flags 1 --mergeeltwise 0 --mergeconcat 0  --stopquantNodes /model.24/Reshape /model.24/Reshape_1 /model.24/Reshape_2 --avoidnode /model.24/dfl/conv/Conv

