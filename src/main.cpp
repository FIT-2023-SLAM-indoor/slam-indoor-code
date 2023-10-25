#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdio>

#include "cameraCalibration.h"

using namespace cv;


int main(int argc, char** argv )
{
    Mat cameraMatrix;
    // Choose method for calibration using CalibrationOption enum
    calibration(cameraMatrix, CalibrationOption::configureFromVideo);
    std::cout << cameraMatrix;
    return 0;
}