#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"
#include "bestFittingPlane.h"

using namespace cv;
using namespace std;
void getBestFittingPlaneByPoints(std::vector<Point3f>& points, Point3d& centroid, Mat& normal){
    Mat A = Mat(points.size(),3,CV_32F,points.data());
    transpose(A,A);
    cout<< A <<endl;
    Mat leftSingularVectors, leftSingularValues, transposedMatrixOfRightSingularVectors;
    

    centroid = Point3d(mean(A.row(0))[0],mean(A.row(1))[0],mean(A.row(2))[0]);

    

    A.row(0) = A.row(0) - mean(A.row(0));
    A.row(1) = A.row(1) - mean(A.row(1));
    A.row(2) = A.row(2) - mean(A.row(2));
    cout<< A <<endl;

    SVD::compute(A,leftSingularValues, leftSingularVectors, transposedMatrixOfRightSingularVectors);

    cout << "leftSingularVectors:" << endl;
    cout << leftSingularVectors << endl << endl;

    cout << "leftSingularValues:" << endl;
    cout << leftSingularValues << endl << endl;
    
    normal = Mat(leftSingularVectors.col(2));
}
int test() {
    vector<Point3f> points;
    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(70.0, 60.0, 10.0));
    points.push_back(Point3f(5.0, 90.0, 10.0));
    points.push_back(Point3f(60.0, 90.0, 10.0));

    points.push_back(Point3f(5.0, 60.0, 100.0));
    points.push_back(Point3f(70.0, 60.0, 100.0));
    points.push_back(Point3f(5.0, 90.0, 100.0));
    points.push_back(Point3f(60.0, 90.0, 100.0));

    points.push_back(Point3f(75.0, 60.0, 10.0));
    points.push_back(Point3f(140.0, 60.0, 10.0));
    points.push_back(Point3f(75.0, 90.0, 10.0));
    points.push_back(Point3f(130.0, 90.0, 10.0));


    
   
    vector<Vec3b> colors;
    

    viz::Viz3d window = makeWindow();
    viz::WCloud cloudWidget = getPointCloudFromPoints(points,colors);
    Mat normal;
    Point3d centroid;
    getBestFittingPlaneByPoints(points,centroid,normal);
    cout<< "normal: " <<endl;
    cout<< normal  <<endl;
    cout<< "centroid:" << centroid << endl;
    viz::WPlane bestFittingPlane(centroid,normal,Vec3d(1,1,1),Size2d(Point2d(100,100)));
    
    cloudWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    window.showWidget("point_cloud", cloudWidget);
    window.showWidget("coordinate", viz::WCoordinateSystem(100));
    window.showWidget("bestPlane",bestFittingPlane);
    startWindowSpin(window);

    return 0;
}
