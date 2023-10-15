#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "poseEstimation.h"

using namespace cv;

int main(int argc, char** argv )
{
    cv::Mat q = (cv::Mat_<double>(2, 2) << 1, 2, 1, 3);
    cv::Mat g = (cv::Mat_<double>(2, 2) << 2, 3, 2, 4);
    countMatricies(q, g);
    return 0;
}