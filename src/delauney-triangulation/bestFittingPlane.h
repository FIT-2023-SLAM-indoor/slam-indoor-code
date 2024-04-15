#include <iostream>
#include <opencv2/opencv.hpp>
#pragma once

using namespace std;
using namespace cv;

void getBestFittingPlaneByPoints(std::vector<Point3f>& points, Point3f& centroid, Mat& normal);
int test();
