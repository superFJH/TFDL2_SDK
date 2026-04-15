export LD_LIBRARY_PATH=$(pwd)/../../lib:$(pwd)/../../lib/CV_NPU40T

./build/Classification ./cat.jpg ./synset_words.txt  ./runconfig.json ./ResNet50.fb