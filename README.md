# slam-indoor-code
Repository with source code of SLAM indoor project

## Requirements
- OpenCV 4.8.0 
- CMake 3.26.2 ([release](https://github.com/Kitware/CMake/releases/tag/v3.27.6))


## Environment configuration
### OpenCV with libs installation
Короче, много всего попробовал точную последовательность не помню, напишу примерно
Использовал две статьи:
1. Для установки VIZ https://habr.com/ru/companies/intel/articles/217021/
просят libvtk5-dev но щас только libvtk7-dev
```sh
sudo apt-get install libvtk7-dev
```
2. Теперь ко второй статье https://bksp.space/blog/en/2020-03-01-compiling-opencv-with-structure-from-motion-module-python-bindings.html
```sh
sudo apt install build-essential
sudo apt install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
#тут некоторые библиотеки могли не установится но пофиг
sudo apt install libeigen3-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-
#тут вроде всё должно сработать
#теперь я не знаю, можно ли установить всё поверх нашего установщика, поэтому сделаем новый билд
mkdir opencv_build
cd opencv_build
#дальше просят установить керас, снизу есть гайд, как это сделать, но вот тут возможно лучше воспользовать гайдом из статьи, потому что сфм какого-то фига не видит керас поэтому пришлось вручную писать дефан что керас есть
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
#тут в статье предлагают менять названия реконстрактов в сфм потому что там 4 одинаковых но я ничего не менял
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules -D BUILD_SHARED_LIBS=ON -D BUILD_opencv_sfm=ON -D OPENCV_ENABLE_NONFREE=ON -D BUILD_SHARED_LIBS=ON  -D BUILD_TESTS=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -D WITH_FFMPEG=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_VTK=ON ..
#Объединение флагов первых двух статей и нашей инструкции к установки opencv 
make -j8
sudo make install
```



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
