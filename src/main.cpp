#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdio>

#include "poseEstimation.h"
#include "cameraCalibration.h"

using namespace cv;

int main(int argc, char** argv )
{
    // cv::Mat q = (cv::Mat_<double>(2, 2) << 1, 2, 1, 3);
    // cv::Mat g = (cv::Mat_<double>(2, 2) << 2, 3, 2, 4);
    // countMatricies(q, g);
    VideoCapture cap("./data/for_calib.mp4");
//    VideoCapture cap(0);
    chessboardCalibration(cap, 15, 5);
    Mat cameraMatrix;
    loadCalibration("./config/cameraMatrix.xml", cameraMatrix);
    std::cout << cameraMatrix;
    return 0;
}