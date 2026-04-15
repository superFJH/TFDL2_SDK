export LD_LIBRARY_PATH=$(pwd)/../../lib:$(pwd)/../../lib/CV_NPU40T

./build/Cityscape $1 ./runconfig.json $2