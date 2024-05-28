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
double distance(Point2f& p1, Point2f& p2);

double distance(Point3f& p1, Point3f& p2);

void projectPointOnPlane(
    Point3f& point,
    Vec3d& normal,
    Point3f& centroid,
    Point3f& projectedPoint);

bool insideCircum(Point2f& point, Triangle& triangle);

bool isPointInVector(Point2f& pt, std::vector<Point2f>& points);

double sqr(double x);

void getCircumByTriangle(Triangle& Triangle, double& radius, Point2f& center);

void getLineByTwoPoints(Point2f& point1,Point2f& point2,double& k, double& m);

void clusterizePoints(std::vector<cv::Point3f>& points,
std::vector<cv::Vec3b>& colors,
std::vector<std::vector<int>>& comps);

void findComps(cv::Mat& graph, int size,std::vector<std::vector<int>>& comps);

void dfs(int index,int size,std::vector<int>& comp,std::vector<bool>& used,cv::Mat& graph);