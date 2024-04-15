#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"
#include "geomAdditionalFunc.h"
#include "bestFittingPlane.h"


using namespace cv;
using namespace std;
void getBestFittingPlaneByPoints(std::vector<Point3f>& points, Point3f& centroid, Vec3d& normal){
    Mat A = Mat(points.size(),3,CV_32F,points.data());
    transpose(A,A);
    cout<< A <<endl;
    Mat leftSingularVectors, leftSingularValues, transposedMatrixOfRightSingularVectors;
    

    centroid = Point3f(mean(A.row(0))[0],mean(A.row(1))[0],mean(A.row(2))[0]);

    

    A.row(0) = A.row(0) - mean(A.row(0));
    A.row(1) = A.row(1) - mean(A.row(1));
    A.row(2) = A.row(2) - mean(A.row(2));
    

    SVD::compute(A,leftSingularValues, leftSingularVectors, transposedMatrixOfRightSingularVectors);

    cout << "leftSingularVectors:" << endl;
    cout << leftSingularVectors << endl << endl;

    cout << "leftSingularValues:" << endl;
    cout << leftSingularValues << endl << endl;
     normal = Vec3d(
        leftSingularVectors.col(2).at<double>(0,0),
        leftSingularVectors.col(2).at<double>(1,0),
        leftSingularVectors.col(2).at<double>(2,0)
     );
   
}
int test() {
    vector<Point3f> points;
    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(70.0, 60.0, 10.0));
    points.push_back(Point3f(5.0, 90.0, 10.0));
    points.push_back(Point3f(60.0, 90.0, 10.0));


    points.push_back(Point3f(95.0, 60.0, 10.0));
    points.push_back(Point3f(150.0, 60.0, 12.0));

    points.push_back(Point3f(95.0, 90.0, 10.0));
    points.push_back(Point3f(160.0, 90.0, 10.0));




    
   
    vector<Vec3b> colors;
    

    viz::Viz3d window = makeWindow();
    viz::WCloud cloudWidget = getPointCloudFromPoints(points,colors);
    Vec3d normal;
    Point3f centroid;
    getBestFittingPlaneByPoints(points,centroid,normal);
    cout<< "normal: " <<endl;
    cout<< normal  <<endl;
    cout<< "centroid:" << centroid << endl;
    viz::WPlane bestFittingPlane(centroid,normal,Vec3d(1,1,1),Size2d(Point2d(150,150)));
    std::vector<Point3f> projectedPoints;
    for (int i = 0;i < points.size();i++){
        Point3f projectedPoint;
        projectPointOnPlane(points.at(i),normal,centroid,projectedPoint);
        projectedPoints.push_back(projectedPoint);

        
    }
    std::vector<Vec3b> projColors;
    for (int i = 0;i< projectedPoints.size();i++){
        projColors.push_back(viz::Color::blue());
    }
    viz::WCloud projectedPointsWidget = getPointCloudFromPoints(projectedPoints,projColors);
    projectedPointsWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    cloudWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    window.showWidget("point_cloud", cloudWidget);
    window.showWidget("point_cloud2", projectedPointsWidget);
    window.showWidget("coordinate", viz::WCoordinateSystem(100));
    window.showWidget("bestPlane",bestFittingPlane);
    startWindowSpin(window);

    return 0;
}