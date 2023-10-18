#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "poseEstimation.h"
#include "cameraCalibration.h"


int main(int argc, char** argv )
{
    // cv::Mat q = (cv::Mat_<double>(2, 2) << 1, 2, 1, 3);
    // cv::Mat g = (cv::Mat_<double>(2, 2) << 2, 3, 2, 4);
    // countMatricies(q, g);
    cv::VideoCapture cap("./data/for_calib.mp4");
    calibrate(cap);
    return 0;
}