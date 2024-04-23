#include <iostream>
#include <opencv2/opencv.hpp>
#include "bestFittingPlane.h"
#include "geomAdditionalFunc.h"
using namespace cv;
double sqr(double x){
    return x*x;
}


double distance(Point2f& p1, Point2f& p2){
    return sqrt(sqr((p1 - p2).x) + sqr((p1 - p2).y));
}

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

bool insideCircum(Point2f& point, Triangle& triangle){
    double radius;
    Point2f center;
    getCircumByTriangle(triangle,radius,center);
    return distance(point,center) <= radius;
}

bool isPointInVector(Point2f& pt, std::vector<Point2f>& points){
    for (int i = 0;i < points.size();i++){
        if (pt.x == points.at(i).x && pt.y == points.at(i).y){
            return true;
        }
    }
    return false;
}


void getCircumByTriangle(Triangle& triangle, double& radius, Point2f& center){
    double k1;
    double k2;
    double m;

    int choosenPoint1;
    int choosenPoint2;
    int lastPoint;
    if (abs((triangle.points.at(1) - triangle.points.at(0)).y) > 0.001){
        choosenPoint1 = 0;
        choosenPoint2 = 1;
        lastPoint = 2;
        if ( abs((triangle.points.at(1) - triangle.points.at(2)).y) < 0.001){
            choosenPoint1 = 2;
            choosenPoint2 = 0;
            lastPoint = 1;
        }
    }else {
        choosenPoint1 = 1;
        choosenPoint2 = 2;
        lastPoint = 0;
    }
    
    getLineByTwoPoints(triangle.points.at(choosenPoint1),triangle.points.at(choosenPoint2),k1,m);
    getLineByTwoPoints(triangle.points.at(lastPoint),triangle.points.at(choosenPoint2),k2,m);


    Point2f p1 = (triangle.points.at(choosenPoint2) - triangle.points.at(choosenPoint1))/ 2 + triangle.points.at(choosenPoint1);
    Point2f p2 = (triangle.points.at(choosenPoint2) - triangle.points.at(lastPoint))/ 2 + triangle.points.at(lastPoint);

    cout << "Point1:" <<  p1 << endl; 
    cout << "Point2:" <<  p2 << endl; 

    double m1 = p1.y + 1/k1 * p1.x;
    double m2 = p2.y + 1/k2 * p2.x;

    cout << "k1:" <<  k1 << endl; 
    cout << "k2:" <<  k2 << endl; 
    
    cout << "m1:" <<  m1 << endl; 
    cout << "m2:" <<  m2 << endl; 

    double x = (m2 - m1)/ (1/k2 - 1/k1);
    double y = m1 - (1/k1)*x;

    center = Point2f(x,y);
    radius = distance(triangle.points.at(0), center);

}
void getLineByTwoPoints(Point2f& point1,Point2f& point2,double& k, double& m){
    k = (point2.y - point1.y) / (point2.x - point1.x);
    m = (point2.x * point1.y - point2.y* point1.x) / (point2.x - point1.x);
}

