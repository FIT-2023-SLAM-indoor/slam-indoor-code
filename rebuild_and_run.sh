#!/bin/sh
cmake . -B build
make -C build -j8
./build/slam-indoor-code $1
