#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"
#include "geomAdditionalFunc.h"
#include <algorithm>
#include "bowyerWatson.h"
using namespace cv;

void triangulation(std::vector<Point2f>& points,std::vector<Triangle>& triangulation){
    assert(triangulation.size() == 0);
    Triangle super;
    Point2f superFirst = Point2f(0,1000);
    Point2f superSecond = Point2f(1000,-1000);
    Point2f superThird = Point2f(-1000,-1000);
    super.points.push_back(superFirst);
    super.points.push_back(superSecond);
    super.points.push_back(superThird);

    triangulation.push_back(super);
    for (int i = 0;i< points.size();i++){
        Point2f point = points.at(i);
        std::vector<int> indexes;

        std::vector<Triangle> badTriangles;
        for (int j = 0;j < triangulation.size();j++){
            Triangle triangle = triangulation.at(j);
            if( insideCircum(point, triangle)){
                badTriangles.push_back(triangle);
                indexes.push_back(j);
            }
        }
        std::vector<Edge> polygon;
        for (int triangleIndex = 0; triangleIndex < badTriangles.size();triangleIndex ++){
            for (int k = 0; k < 3; k++){
                Point2f first = badTriangles.at(triangleIndex).points.at(k);
                Point2f second = badTriangles.at(triangleIndex).points.at((k+1)%3);
                bool answer  = true;

                for (int otherTrianglesIndex = 0;otherTrianglesIndex < badTriangles.size(); otherTrianglesIndex++){
                    if (otherTrianglesIndex != triangleIndex){
                        Triangle other = badTriangles.at(otherTrianglesIndex);
                        if (isPointInVector(first,other.points) && isPointInVector(second,other.points)){
                            answer = false;
                        }
                    }
                }
                if (answer){
                    Edge edge;
                    edge.start = first;
                    edge.end = second;
                    polygon.push_back(edge);
                }
            }
        }

        for (int triangleIndex = 0; triangleIndex <badTriangles.size();triangleIndex++){
            triangulation.erase(triangulation.begin() + (indexes.at(triangleIndex) - triangleIndex));
            
        }
        for (int i = 0;i< polygon.size();i++){
            Triangle newTri;
            newTri.points.push_back(polygon.at(i).start);
            newTri.points.push_back(polygon.at(i).end);
            newTri.points.push_back(point);
            triangulation.push_back(newTri);
        }
    
    }
    for (int i = 0;i<triangulation.size();i++){
        Triangle curr = triangulation.at(i);
        if (isPointInVector(superFirst,curr.points) ||
            isPointInVector(superSecond,curr.points) ||
            isPointInVector(superThird,curr.points)
        ){
            triangulation.erase(triangulation.begin() + i);
            i--;
        }
    }

}