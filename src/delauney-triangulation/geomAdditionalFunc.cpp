#include <iostream>
#include <opencv2/opencv.hpp>
#include "bestFittingPlane.h"
using namespace cv;
void projectPointOnPlane(Point3f& point, Mat& normal, Point3f& centroid, Point3f& projectedPoint){
    double x = 
        ((point.x - centroid.x)*normal.at<double>(0,0) + 
        (point.y - centroid.y)*normal.at<double>(1,0) + 
        (point.z - centroid.z)*normal.at<double>(2,0))/
     (normal.at<double>(0,0) * normal.at<double>(0,0) + 
      normal.at<double>(1,0) *normal.at<double>(1,0) +
        normal.at<double>(2,0) *normal.at<double>(2,0));

    projectedPoint.x = point.x - normal.at<double>(0,0)*x;
    projectedPoint.y = point.y - normal.at<double>(1,0)*x;
    projectedPoint.z = point.z - normal.at<double>(2,0)*x;
}

