export LD_LIBRARY_PATH=$(pwd)/../../lib:$(pwd)/../../lib/CV_NPU10T

./build/yolov5 $1 ./runconfig.json ./4271749545783.jpg
