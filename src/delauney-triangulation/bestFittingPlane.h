#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#pragma once

using namespace std;
using namespace cv;

void getBestFittingPlaneByPoints(std::vector<Point3f>& points, Point3f& centroid, Vec3d& normal);
int test();
viz::WMesh makeMesh(vector<Point3f>& points, vector<Vec3b>& colors);