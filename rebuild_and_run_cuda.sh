#!/bin/sh
cmake . -B cuda_build -D USE_CUDA=YES
make -C cuda_build -j8
./build/slam-indoor-code $1