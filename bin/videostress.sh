#export LD_LIBRARY_PATH=$(pwd):$(pwd)/../lib:$(pwd)/../lib/opencv
export LD_LIBRARY_PATH=$(pwd):/zhome/fjh/tmp/NPU40T/lib/
#./VideoStress $1 $2 /zhome/fjh/tmp/03000002808000000.mp4 /zhome/fjh/tmp/03000002808000000.mp4 #rtsp://admin:Root1234@176.16.0.230:554/streaming/channels/101 rtsp://admin:Root1234@176.16.0.230:554/streaming/channels/201 #rtsp://10.10.12.65:554/factory2.265 rtsp://10.10.12.65:554/factory2.265

./VideoStress $1 $2 rtsp://admin:Root1234@176.16.0.230:554/streaming/channels/101 rtsp://admin:Root1234@176.16.0.230:554/streaming/channels/201 #rtsp://10.10.12.65:554/factory2.265 rtsp://10.10.12.65:554/factory2.265
