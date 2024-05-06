#include <iostream>
#include <opencv2/opencv.hpp>
#include "../vizualizationModule.h"
#include "geomAdditionalFunc.h"
#include "bestFittingPlane.h"
#include "bowyerWatson.h"
#include <utility>

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
    vector<pair<Point2f,Point3f>> pairs;
    vector<Point3f> points;
    
    /*
    points.push_back(Point3f(0.0, 0.0, 0.0));
    points.push_back(Point3f(70.0, 0.0, 0.0));
    points.push_back(Point3f(70.0, 50.0, 0.0));
    points.push_back(Point3f(70.0, 50.0, 70.0));

    points.push_back(Point3f(0.0, 49.0, 70.0));
    points.push_back(Point3f(0.0, 0.0, 70.0));
    points.push_back(Point3f(0.0, 50.0, 0.0));
    points.push_back(Point3f(70.0, 0.0, 70.0));*/
    srand(time(0));
    for (int i = 0;i< 1000;i++){
        int x = rand()%3000;
        int y = 100+ rand()%300 - rand()%300;
        int z = rand()%3000;
        points.push_back(Point3f(x,y,z));
    }

    

    
   
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
        //projectedPoints.push_back(projectedPoint);
        
        projectedPoint = projectedPoint - centroid;


        double coef = sqrt(sqr(projectedPoint.y) / (sqr(projectedPoint.x) + sqr(projectedPoint.z)) +1);


        projectedPoint = projectedPoint*coef;
        
        projectedPoint.y = 0;
        //cout << projectedPoint <<endl;

        projectedPoints.push_back(projectedPoint);

        

        
    }
    
    std::vector<Point2f> pts;
    for (int i = 0;i< projectedPoints.size();i++){
        Point2f cur  = Point2f(projectedPoints.at(i).x,projectedPoints.at(i).z);
        pts.push_back(cur);
        pairs.push_back(pair(cur,points.at(i)));
    }
    vector<Vec3b> proj2Colors;
    for (int i = 0;i< projectedPoints.size();i++){
        proj2Colors.push_back(viz::Color::celestial_blue());
    }
    viz::WCloud projectedPointsWidget = getPointCloudFromPoints(projectedPoints,proj2Colors);
    //projectedPointsWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    window.showWidget("point_cloud22", projectedPointsWidget);



    cloudWidget.setRenderingProperty( cv::viz::POINT_SIZE, 10);
    window.showWidget("point_cloud", cloudWidget);
    window.showWidget("coordinate", viz::WCoordinateSystem(100));
    //window.showWidget("bestPlane",bestFittingPlane);

    
    int maxDistance = 30000;
    
    



    

    vector<Vec3b> projColors;

    vector<Triangle> triang;
    triangulation(pts,triang);
    cout << "Triangulation:" << endl;
    for (int i = 0;i< triang.size();i++){
        bool flag = true;

        cv::Mat polygon3 = (cv::Mat_<int>(1,4) << 3, 0, 1, 2);
        vector<int> faces{3, 0, 1, 2};
        vector<Point3f> cloud3;
        cout << triang.at(i).points << endl;
        for (int j = 0;j < 3;j++){
            Point3f pt;
            for (int k = 0;k< pairs.size();k++){
                if (pairs.at(k).first.x == triang.at(i).points.at(j).x && 
                pairs.at(k).first.y == triang.at(i).points.at(j).y){
                    pt = pairs.at(k).second;
                    break;
                }

            }
            cloud3.push_back(pt);
            if (j > 0 && distance(cloud3.at(j),cloud3.at(j-1)) > maxDistance){
                flag = false;
            }
        }
        
        if (!flag){
            continue;
        }
        cv::viz::WMesh trWidget(cloud3, faces);

        trWidget.setColor(viz::Color::indigo());
        trWidget.setRenderingProperty(viz::OPACITY, 0.005 * (i+1));
        trWidget.setRenderingProperty(viz::SHADING, viz::SHADING_FLAT);
        trWidget.setRenderingProperty(viz::REPRESENTATION, viz::REPRESENTATION_SURFACE);
        String str = to_string(i);
        window.showWidget(str, trWidget);
    }


    cout << "Triangulation ended" << endl;

    vector<Point3f> trPts;
    for (int i = 0;i< pts.size();i++){
        trPts.push_back(Point3f(pts.at(i).x,0,pts.at(i).y));
    }

    viz::WCloud trPtsWidget = getPointCloudFromPoints(trPts,projColors);
    trPtsWidget.setRenderingProperty( cv::viz::POINT_SIZE, 5);
    //window.showWidget("pts",trPtsWidget);
    startWindowSpin(window);


    return 0;
}
