#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdio>

#include "cameraCalibration.h"
#include "fastExtractor.h"

using namespace cv;


int main(int argc, char** argv)
{
    Mat image, result;
    std::vector<KeyPoint> features;

    // Saved the image into an N-dimensional array
    image = imread("data/arfcts/1.png");  // ImreadModes::IMREAD_GRAYSCALE
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }

    // Applied the FAST algorithm to the image and saved the image
    // with the highlighted features in @result
    fastExtractor(image, features);

    drawKeypoints(image, features, result);

    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", result);
    // Each image displays for 4 seconds
    waitKey(4000);
  
    Mat cameraMatrix;
    // Choose method for calibration using CalibrationOption enum
    calibration(cameraMatrix, CalibrationOption::configureFromFiles);
    std::cout << cameraMatrix;
    return 0;
}