#include <iostream>
#include <opencv2/opencv.hpp>
#include "bestFittingPlane.h"
using namespace cv;
void projectPointOnPlane(Point3f& point, Vec3d& normal, Point3f& centroid, Point3f& projectedPoint){
    double x = 
        ((point.x - centroid.x)*normal[0] + 
        (point.y - centroid.y)*normal[1] + 
        (point.z - centroid.z)*normal[2])/
     (normal[0] * normal[0] + 
     normal[1]*normal[1] +
        normal[2] *normal[2]);

    projectedPoint.x = point.x - normal[0]*x;
    projectedPoint.y = point.y - normal[1]*x;
    projectedPoint.z = point.z - normal[2]*x;
}

