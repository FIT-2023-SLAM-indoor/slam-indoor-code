#!/bin/sh
mkdir build || rm ./build -r -f
cmake . -B build
make -C build -j8
./build/slam-indoor-code $1