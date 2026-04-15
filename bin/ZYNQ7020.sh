#!/bin/bash

# 检查是否为 Ubuntu 系统
if [ ! -f /etc/os-release ]; then
    echo "错误：非 Ubuntu/Debian 系统！"
    exit 1
fi

# 读取系统版本信息
source /etc/os-release

# 检查是否为 Ubuntu
if [ "$ID" != "ubuntu" ]; then
    echo "错误：非 Ubuntu 系统！"
    exit 1
fi

# 提取主版本号和次版本号（如 18.04 → 18 和 04）
MAJOR_VERSION=$(echo "$VERSION_ID" | cut -d '.' -f 1)
MINOR_VERSION=$(echo "$VERSION_ID" | cut -d '.' -f 2)

# 判断版本
if [ "$MAJOR_VERSION" -eq 18 ] && [ "$MINOR_VERSION" -eq 04 ]; then
    echo "系统是 Ubuntu 18.04 (Bionic Beaver) 猜测是NPU10T芯片"
    export LD_LIBRARY_PATH=$(pwd)/../lib:$(pwd)/../lib/CV_NPU10T
elif [ "$MAJOR_VERSION" -gt 18 ] || \
     { [ "$MAJOR_VERSION" -eq 18 ] && [ "$MINOR_VERSION" -ge 04 ]; }; then
    echo "系统是 Ubuntu 18.04 或更高版本（当前: $VERSION_ID） 猜测是NPU40T芯片"
    export LD_LIBRARY_PATH=$(pwd)/../lib:$(pwd)/../lib/CV_NPU40T
else
    echo "系统版本低于 18.04（当前: $VERSION_ID） 退出"
    exit 1
fi

#export TFNN_AVAILABLE_CORES=8,9,10,11,12,13,14,15
#gdb --args ./WrapNetTest data/frame_00000600.jpg data/frame_00000600_ref.jpg ./ -1 1 4 data/model/dev_E97_int8.fb
#./MaskRCNNTEST data/frame_00000800.jpg data/frame_00000600_ref.jpg ./ -1 0 5 ../Python/R04.fb
#./WrapNetTest data/frame_00000600.jpg data/frame_00000600_ref.jpg ./ -1 1 5 data/model/dev_E97_int8.fb
#./classification data/cat.jpg data/synset_words.txt 1 1 5 data/model/mobilenetV1_q.fb
#./classification data/cat.jpg data/synset_words.txt 1 1 5 data/model/mobilenetv1_int8.fb
#./classification data/cat.jpg data/synset_words.txt 1 1 5 data/model/mobilenetv2.fb
#./classification $1 data/synset_words.txt 0 1 5 runconfig.json empty_modify.json data/model/efficientnet-b0.fb
#echo
#./classification data/cat.jpg data/synset_words.txt 1 1 5 runconfig.json modify.json $1
#./TestImgenet $1 data/synset_words.txt 0 1 5 runconfig.json $2 data/model/efficientnet-b0_entro_int8.fb
#./ImageNetServer 4 $1
#./classification data/cat.jpg data/synset_words.txt  1 5 hat.fb
#./Benchmark 5 data/model/mbl-ssd_int8.fb 2 1000 1 runconfig.json
#./Benchmark 5 HF_c57_blob_int8.fb 4 1000 0
#./Benchmark 5 ../Python/cloth_int8.fb 2 1000 1
#./Benchmark 5 data/model/Resnet50_int8.fb 4 1000 0
#numactl --membind=1 --cpunodebind=1 ./Benchmark $1 8 2000 $2 runconfig.json data/cat.jpg $3
./Benchmark $1 8 2000 $2 runconfig.json data/cat.jpg $3
#./TFConvertor --proto 5 --quantize 1 --calibration_mode 0 --calibration_list imglist --config runconfig.json --input_file data/model/yolov5-common.fb --output data/model/yolov5-commonQ --scale 0.00392157 0.00392157 0.00392157 --mean 0 0 0 --cv_flags 1 --mergeeltwise 0 --mergeconcat 0  --stopquantNodes /model.24/Sigmoid /model.24/Sigmoid_1 /model.24/Sigmoid_2 #/model.21/Reshape /model.21/Reshape_1 /model.21/Reshape_2  --avoidnode /model.21/dfl/conv/Conv
#./TFConvertor --proto 5 --quantize 1 --calibration_mode 0 --calibration_list imglist --config runconfig.json --input_file data/model/yolo12m.fb --output data/model/yolo12mQ --scale 0.00392157 0.00392157 0.00392157 --mean 0 0 0 --cv_flags 1 --mergeeltwise 0 --mergeconcat 0 --avoidnode /model.21/dfl/conv/Conv --stopquantNodes /model.21/Reshape /model.21/Reshape_1 /model.21/Reshape_2 
#./TFConvertor --proto 5 --quantize 1 --calibration_mode 0 --calibration_list data/blur.list --config runconfig.json --input_file data/model/MIMO-UNetPlus.fb --output data/model/MIMO-UNetPlusQ_10T --scale 0.00392157 0.00392157 0.00392157 --mean 0 0 0 --cv_flags 1 --mergeeltwise 1 --mergeconcat 1
#./Benchmark
#./heatmapDemo data/OmniPlayerSnapshot-2021-02-22-16h54m42s128.png data/model/baidu_ocr_q.fb runconfig.json
#gdb --args ./App script/SemanticSeg.json
#gdb --args ./App script/Yolov4Detection.json
#./App script/CharaDetect.json
#./App script/SSDDetection.json
#./App script/Detection.json
#./App script/detect\&keypoint2.json
#./App script/detect\&keypoint.json
#./App script/PlateRecognizer.json
#./App $1
#gdb --args ./PlateDetect $1
#./twocorebenchmark $1 3 100 runconfig.json
#./OpenPose data/ski.jpg data/model/pose_body25_int8.fb 1
#./Detection data/model/Yolov4_int8.fb runconfig.json rtsp://192.168.1.43:8554/Stream 0 0.5 1 modify.json outdd
#./Detection data/model/Yolov4_int8.fb runconfig.json data/000335.png 0 0.5 0 modify.json outdd
#./Detection data/model/hourglasshand_relu6_256.fb runconfig.json $1 0 0.5 5 modify.json outdd
#./tfdl_yolov3 data/model/Yolov3_int8.fb runconfig.json $1 0 $2 1 modify.json $3
#./tfdl_yolov3 data/model/Yolov4_test2.fb runconfig.json rtsp://192.168.1.43:8554/video 0 $1 1 modify.json outdd
#./simpletest conv 10
#./TestModule
#./TFApp $1
#valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./TestModule
#./Benchmark 5 data/model/yolov3-voc_aqm.tfdl 2 1000 1 runconfig.json
#./tfdl_yolov3 data/model/yolov3-voc_aqm.tfdl runconfig.json data/helmet.jpg 1
#./testtvm
#valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all  ./MemleakTest 5 data/model/mobilenetSSD_int8.fb 1 1
#./MemleakTest 5 data/model/mobilenetSSD_int8.fb 1 1
#./classification data/apple.png data/cifar100.txt 0 1 5 data/model/oct_int8.fb
#./SSDDetect ../../img000046.jpg ./ 1 5 HF_c57_blob_int8.fb
#./TFConvertor --help
#./TFConvertor --proto 2 --input_file data/model/mobilenet_v1_1.0_224_frozen.pb --debug 1 --output mobilenetv1
#./TFConvertor --proto 1 --input_file ../../model/Caffe/HF_c57_blob.prototxt --model ../../model/Caffe/HF_c57_blob.model --debug 1 --output HF_c57_blob
#./TFConvertor --proto 5 --input_file HF_c57_blob.fb --debug 1 --output HF_c57_blob_int8 --quantize 1 --calibration_mode 0 --calibration_list ../../cali.list --config runconfig.json --cv_flags 0
#./BatchTest config.json
#./Benchmark 5 hat.fb 2 1000 1
