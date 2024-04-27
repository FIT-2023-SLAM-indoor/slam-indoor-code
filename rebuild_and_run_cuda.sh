#!/bin/sh
mkdir build || rm ./build -r -f
cmake . -B build -D USE_CUDA=YES
make -C build -j8
./build/slam-indoor-code $1