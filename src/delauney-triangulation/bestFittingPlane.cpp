#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"

using namespace cv;
using namespace std;
int test() {
    vector<Point3f> points;
    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(70.0, 60.0, 10.0));
    points.push_back(Point3f(5.0, 70.0, 10.0));
    points.push_back(Point3f(60.0, 70.0, 10.0));

   
    vector<Vec3b> colors;
    Mat A = Mat(points.size(),3,CV_32F,points.data());
    transpose(A,A);
    cout<< A <<endl;
    Mat leftSingularVectors, leftSingularValues, transposedMatrixOfRightSingularVectors;
    

    Point3d centroid(mean(A.row(0))[0],mean(A.row(1))[0],mean(A.row(2))[0]);
    cout<< "centroid:" << centroid << endl;

    A.row(0) = A.row(0) - mean(A.row(0));
    A.row(1) = A.row(1) - mean(A.row(1));
    A.row(2) = A.row(2) - mean(A.row(2));
    cout<< A <<endl;

    SVD::compute(A,leftSingularValues, leftSingularVectors, transposedMatrixOfRightSingularVectors);

    cout << "leftSingularVectors:" << endl;
    cout << leftSingularVectors << endl << endl;

    cout << "leftSingularValues:" << endl;
    cout << leftSingularValues << endl << endl;

    viz::Viz3d window = makeWindow();
    viz::WCloud cloudWidget = getPointCloudFromPoints(points,colors);
    cout<< "normal: " <<endl;
    cout<< leftSingularVectors.col(2) <<endl;

    

    viz::WPlane bestFittingPlane(centroid,leftSingularVectors.col(2),Vec3d(1,1,1),Size2d(Point2d(100,100)));
    cloudWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    window.showWidget("point_cloud", cloudWidget);
    window.showWidget("coordinate", viz::WCoordinateSystem(100));
    window.showWidget("bestPlane",bestFittingPlane);
    startWindowSpin(window);

    return 0;
}