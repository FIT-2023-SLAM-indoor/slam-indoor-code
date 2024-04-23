#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"
#include "geomAdditionalFunc.h"
#include "bestFittingPlane.h"
#include "bowyerWatson.h"


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
    /*
    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(70.0, 40.0, 50.0));
    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(60.0, 120.0, 50.0));

    points.push_back(Point3f(25.0, 60.0, 10.0));
    points.push_back(Point3f(90.0, 40.0, 50.0));
    points.push_back(Point3f(25.0, 60.0, 10.0));
    points.push_back(Point3f(80.0, 120.0, 50.0));
    
    points.push_back(Point3f(5.0, 80.0, 30.0));
    points.push_back(Point3f(70.0, 80.0, 70.0));
    points.push_back(Point3f(5.0, 100.0, 30.0));
    points.push_back(Point3f(60.0, 160.0, 70.0));

    points.push_back(Point3f(25.0, 70.0, 30.0));
    points.push_back(Point3f(90.0, 40.0, 70.0));
    points.push_back(Point3f(25.0, 20.0, 30.0));
    points.push_back(Point3f(80.0, 10.0, 70.0));
    */

    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(70.0, 40.0, 50.0));
    points.push_back(Point3f(5.0, 60.0, 10.0));
    points.push_back(Point3f(60.0, 120.0, 50.0));




    
   
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
        
        projectedPoint = projectedPoint - centroid;


        double coef = sqrt(sqr(projectedPoint.y) / (sqr(projectedPoint.x) + sqr(projectedPoint.z)) +1);


        projectedPoint = projectedPoint*coef;
        
        projectedPoint.y = 0;
        cout << projectedPoint <<endl;

        projectedPoints.push_back(projectedPoint);

        

        
    }
    std::vector<Vec3b> projColors;
    for (int i = 0;i< projectedPoints.size();i++){
        projColors.push_back(viz::Color::blue());
    }
    //viz::WCloud projectedPointsWidget = getPointCloudFromPoints(projectedPoints,projColors);
    //projectedPointsWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    //cloudWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    //window.showWidget("point_cloud", cloudWidget);
    //window.showWidget("point_cloud2", projectedPointsWidget);
    //window.showWidget("coordinate", viz::WCoordinateSystem(100));
    //window.showWidget("bestPlane",bestFittingPlane);

    
    vector<Point2f> pts;
    pts.push_back(Point2f(30,-30));
    pts.push_back(Point2f(30,30));
    pts.push_back(Point2f(0,0));
    pts.push_back(Point2f(50,0));
    pts.push_back(Point2f(70,70));
    pts.push_back(Point2f(80,0));
    pts.push_back(Point2f(80,-80));

 
    



    vector<Triangle> triang;
    triangulation(pts,triang);
    cout << "Triangulation:" << endl;
    for (int i = 0;i< triang.size();i++){
        cv::Mat polygon3 = (cv::Mat_<int>(1,4) << 3, 0, 1, 2);
        vector<int> faces{3, 0, 1, 2};
        vector<Point3d> cloud3;
        cout << triang.at(i).points << endl;
        for (int j = 0;j < 3;j++){
            cloud3.push_back(Point3d(triang.at(i).points.at(j)));
        }


        cv::viz::WMesh trWidget(cloud3, faces);

        trWidget.setColor(viz::Color::indigo());
        trWidget.setRenderingProperty(viz::OPACITY, 0.1 * (i+1));
        trWidget.setRenderingProperty(viz::SHADING, viz::SHADING_FLAT);
        trWidget.setRenderingProperty(viz::REPRESENTATION, viz::REPRESENTATION_SURFACE);
        char str[16]  ="triangnnnn";
        str[0] = i;
        window.showWidget(str, trWidget);
    }


    cout << "Triangulation ended" << endl;
    //getCircumByTriangle(triangle,radius,center);
    //cout << "WTF:" << endl;
    //cout << radius << endl;
    //cout << center << endl;
    //startWindowSpin(window);
    vector<Point3f> trPts;
    for (int i = 0;i< pts.size();i++){
        trPts.push_back(Point3f(pts.at(i)));
    }

    viz::WCloud trPtsWidget = getPointCloudFromPoints(trPts,projColors);
    trPtsWidget.setRenderingProperty( cv::viz::POINT_SIZE, 5);
    window.showWidget("pts",trPtsWidget);
    startWindowSpin(window);


    return 0;
}
