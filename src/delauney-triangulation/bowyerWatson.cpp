#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"
#include "geomAdditionalFunc.h"
#include "bowyerWatson.h"
using namespace cv;

void triangulation(std::vector<Point2d>& points,std::vector<Triangle>& triangulation){
    assert(triangulation.size() == 0);
    Triangle super;
    super.points.push_back(Point2f(0,1000));
    super.points.push_back(Point2f(1000,-1000));
    super.points.push_back(Point2f(-1000,-1000));

    triangulation.push_back(super);
    for (int i = 0;i< points.size();i++){
        Point2f point = points.at(i);
        std::vector<Triangle> badTriangles;
        for (int j = 0;j < triangulation.size();j++){
            Triangle triangle = triangulation.at(j);
            if( insideCircum(point, triangle)){
                badTriangles.push_back(triangle);
            }
        }

    }
}