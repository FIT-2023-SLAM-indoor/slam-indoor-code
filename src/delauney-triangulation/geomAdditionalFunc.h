#include <iostream>
#include <opencv2/opencv.hpp>
#pragma once
using namespace cv;
void projectPointOnPlane(
    Point3f& point,
    Vec3d& normal,
    Point3f& centroid,
    Point3f& projectedPoint);