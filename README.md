# slam-indoor-code
Repository with source code of SLAM indoor project

## Requirements
- OpenCV 4.8.0 
- CMake 3.26.2 ([release](https://github.com/Kitware/CMake/releases/tag/v3.27.6))

## Environment configuration
### OpenCV installation
```sh
sudo apt update
sudo apt install cmake libtbb2 g++ wget unzip ffmpeg libgtk2.0-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libtbb-dev libjpeg-dev libpng-dev libtiff-dev

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
unzip opencv.zip  # files will be extracted to ./opencv-4.8.0
mkdir opencv-build
cd opencv-build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -D WITH_FFMPEG=ON ../opencv-4.8.0  # Make sure FFMPEG and its modules marked "YES"

make -j8  # Number of jobs can be specified
sudo make install
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
  "calibrationPath": "./config/samsung-hp.xml",

  "usePhotosCycle": true,
  "photosPathPattern": "/mnt/c/Users/bakug/YandexDisk/NSU/private/2-1/PAK/static/photos/samsung-tumbochka-s/*.JPG",
  // "videoSourcePath" has an effect when "usePhotosCycle" is false
  "videoSourcePath": "/mnt/c/Users/bakug/YandexDisk/NSU/private/2-1/PAK/static/samsung-hall.mp4",
  
  "outputDataDir": "./data/video_report/video_test",
  
  "useUndistortion": false,

  "requiredExtractedPointsCount": 100,
  "featureExtractingThreshold": 20,

  "framesBatchSize": 1,

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
  "featureMatchingRadius": 100.0,
  "showTrackedPoints": true,

  "RPUseRANSAC": true,
  "RPRANSACProb": 0.999,
  "RPRANSACThreshold": 5.0,
  "RPRequiredGoodPointsPercent": 0.5,
  "RPDistanceTreshold": 200
}
```

---

`./python_utility/config.py`:
```python
PROJECT_PATH = "/PATH/TO/PROJECT/DIR"
VIZ_FILE_PATH = "data/points.txt"
VIZ_PARSE_FORMAT = "xyz"
```
---
So now you can specify program working using configs and run using `./rebuild_and_run.sh` (write `chmod a+x rebuild_and_run.sh` to make this file executable)

### Ceres installation (WIP)