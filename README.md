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
sudo apt install libvtk7-dev
sudo apt install build-essential
sudo apt install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
#тут некоторые библиотеки могли не установится но пофиг
sudo apt install libeigen3-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-
#тут вроде всё должно сработать
```
2. Download sources
```sh
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
unzip opencv.zip  # files will be extracted to ./opencv-4.8.0

wget -O opencv-contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.0.zip
unzip opencv-contrib.zip  # files will be extracted to ./opencv_contrib-4.8.0

mkdir opencv-build
cd opencv-build
```
3. Build & install
```sh
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.8.0/modules/ \
-D BUILD_SHARED_LIBS=ON \
-D BUILD_opencv_sfm=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_SHARED_LIBS=ON \
-D BUILD_TESTS=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=ON \
-D WITH_QT=ON \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D WITH_FFMPEG=ON \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D WITH_VTK=ON \
../opencv-4.8.0/  

# Make sure FFMPEG and its modules marked "YES"

make -j8  # Number of jobs can be specified
sudo make install
```
*For CUDA:*
```sh
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.8.0/modules/ \
-D BUILD_SHARED_LIBS=ON \
-D BUILD_opencv_sfm=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_SHARED_LIBS=ON \
-D BUILD_TESTS=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=ON \
-D WITH_QT=ON \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D WITH_FFMPEG=ON \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D WITH_VTK=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_ARCH_BIN=8.6 \
-D WITH_GSTREAMER=ON \
../opencv-4.8.0/  
```

### Configuration files (may be outdated)
Install `nlohmann-json`:
```bash
sudo apt install nlohmann-json3-dev
```
JSON config (path to file should be specified as command line argument):
```json
{
  "calibrate": false,
  "visualCalibration": true,
  "calibrationPath": "./config/samsung-hv.xml",

  "usePhotosCycle": false,
  "photosPathPattern": "/mnt/c/Users/bakug/YandexDisk/NSU/private/2-1/PAK/static/photos/NSU-hall/*.JPG",
  // "videoSourcePath" has an effect when "usePhotosCycle" is false
  "videoSourcePath": "/mnt/c/Users/bakug/YandexDisk/NSU/private/2-1/PAK/static/samsung-hall.mp4",

  "outputDataDir": "./data/video_report/video_test",

  "useUndistortion": false,

  "requiredExtractedPointsCount": 1000,
  "featureExtractingThreshold": 15,

  "framesBatchSize": 15,

  "requiredMatchedPointsCount": 200,

  "useFeatureTracker": false,
  // This block has an effect when "useFeatureTracker" is true
  "useOwnFeatureTracker": true,
  "FTThreadsCount": 3,
  "useSADOwnFT": true,
  "useSSDOwnFT": false,
  "FTBarrier": 20,
  "FTMaxAcceptableDifference": 2000,

  // This block has an effect when "useFeatureTracker" is false
  "useFM-SIFT-FLANN": true,
  "useFM-SIFT-BF": false,
  "useFM-ORB": false,

  "knnMatcherDistance": 0.7,

  "showTrackedPoints": true,

  // Now this parameters are used only for the first pair of frames
  "RPUseRANSAC": true,
  "RPRANSACProb": 0.999,
  "RPRANSACThreshold": 5.0,
  "RPRequiredGoodPointsPercent": 0.5,
  "RPDistanceThreshold": 200.0,

  "useBundleAdjustment": true,
  "BAHuberLossFunctionParameter": 4.0,
  "BAThreadsCnt": 4
}
```
- `./python_utility/config.py`:
```python
PROJECT_PATH = "/PATH/TO/PROJECT/DIR"
VIZ_FILE_PATH = "data/points.txt"
VIZ_PARSE_FORMAT = "xyz"
```
---
So now you can specify program working using configs and run using `./rebuild_and_run.sh` (write `chmod a+x rebuild_and_run.sh` to make this file executable) // TODO Rework script

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