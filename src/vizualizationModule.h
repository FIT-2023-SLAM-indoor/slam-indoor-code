#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
using namespace cv;
void vizualizePoints(std::vector<Point3f> spatialPoints);

void vizualizePointsAndCameras(
    std::vector<Point3f> spatialPoints,
    std::vector<Mat> rotations, 
    std::vector<Mat> transitions, 
    Mat calibration
    );
