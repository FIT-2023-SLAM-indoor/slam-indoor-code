#include <iostream>
#include <opencv2/opencv.hpp>
#pragma once
using namespace cv;
typedef struct{
    Point2f start;
    Point2f end;
}Edge;

typedef struct{
    std::vector<Point2f> points;
    
}Triangle;


void projectPointOnPlane(
    Point3f& point,
    Vec3d& normal,
    Point3f& centroid,
    Point3f& projectedPoint);

bool insideCircum(Point2f& point, Triangle& triangle);

bool isPointInVector(Point2f& pt, std::vector<Point2f>& points);

double sqr(double x);

void getCircumByTriangle(Triangle& Triangle, double& radius, Point2f& center);

void getLineByTwoPoints(Point2f point1,Point2f point2,double& k, double& m);