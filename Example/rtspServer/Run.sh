#!/bin/bash
export LD_LIBRARY_PATH=$(pwd)/../../lib:$(pwd)/../../lib/CV_NPU10T

./build/RtspServer rtsp://admin:Root1234@176.16.0.230:554/streaming/channels/101 11111