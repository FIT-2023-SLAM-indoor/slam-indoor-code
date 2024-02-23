# slam-indoor-code
Repository with source code of SLAM indoor project

## Requirements
- OpenCV 4.8.0 
- CMake 3.26.2 ([release](https://github.com/Kitware/CMake/releases/tag/v3.27.6))

## Environment configuration
### OpenCV installation
1. Install dependencies and tools:
```sh
sudo apt update
sudo apt install cmake libtbb2 g++ wget unzip ffmpeg libgtk2.0-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libtbb-dev libjpeg-dev libpng-dev libtiff-dev
```
2. Download sources
```sh
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
unzip opencv.zip  # files will be extracted to ./opencv-4.8.0
mkdir opencv-build
cd opencv-build
```
3. Build & install
```sh
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -D WITH_FFMPEG=ON ../opencv-4.8.0  # Make sure FFMPEG and its modules marked "YES"

make -j8  # Number of jobs can be specified
sudo make install
```
### Configuration files (may be outdated)
- `./src/main_config.h`:
```c++
#pragma once

// Uncomment this define to run program for calibration
//#define CALIB

#define CALIBRATION_PATH "./config/samsung-hp.xml"

// Uncomment this define to see found corners through the calibration
//#define VISUAL_CALIB


// Comment this define to process video
#define PHOTOS_CYCLE

#define PHOTOS_PATH_PATTERN "../static/photos/samsung-table-s/*.JPG"

#define VIDEO_SOURCE_PATH "../static/realme-classroom.mp4"

#define OUTPUT_DATA_DIR "./data/video_report/samsung_table_all"

// Comment this define to disable undistortion using calibration data (distortion coefficients are needed)
#define USE_UNDISTORTION

#define REQUIRED_EXTRACTED_POINTS_COUNT 50
#define FEATURE_EXTRACTING_THRESHOLD 20


// For photos cycle this parameter MUST BE 1
#define FRAMES_BATCH_SIZE 1


// Comment this define to use KLT feature-tracker
#define FT_STANDART
#define FT_THREADS_COUNT 3
#define FT_TIME

// Uncomment this define to see frames with tracked points
//#define SHOW_TRACKED_POINTS

// One of this defines MUST BE COMMENTED
//#define FT_SSD
#define FT_SAD

#define FT_BARRIER 20
#define FT_MAX_ACCEPTABLE_DIFFERENCE 20000


// Uncomment this to use RANSAC algorithm for essential matrix estimation
#define USE_RANSAC
#define RANSAC_PROB 0.999
#define RANSAC_THRESHOLD 5.0
#define USE_RANSAC_POINTS_FILTER
#define RANSAC_GOOD_POINTS_PERCENT 0.5


#define RECOVER_POSE_DISTANCE_THRESHOLD 500
#define USE_RECOVER_POSE_POINTS_FILTER
```
- `./python_utility/config.py`:
```python
PROJECT_PATH = "/PATH/TO/PROJECT/DIR"
VIZ_FILE_PATH = "data/points.txt"
VIZ_PARSE_FORMAT = "xyz"
```
---
So now you can specify program working using configs and run using `./rebuild_and_run.sh` (write `chmod a+x rebuild_and_run.sh` to make this file executable)

### Ceres installation
```sh
sudo apt-get install libeigen3-dev libgflags-dev libgoogle-glog-dev

wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz
tar zxf ceres-solver-2.2.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.2.0

make -j3
make test
sudo make install
```
Run for testing installation 
```sh
./bin/simple_bundle_adjuster ../ceres-solver-2.2.0/data/problem-16-22106-pre.txt
```