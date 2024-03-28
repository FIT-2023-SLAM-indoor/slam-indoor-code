#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
using namespace cv;
/*
    Vizualize only points with no trajectory of cameras
    @param spatialPoints points to vizualize
*/
void vizualizePoints(
    std::vector<Point3f> spatialPoints
    );

/*
    Vizualize points with trajectory of camera
    @param spatialPoints points to vizualize
    @param rotations camera rotations vector
    @param transitions camera transitions vector
    @param calibration camera calibration matrix
*/
void vizualizePointsAndCameras(
    std::vector<Point3f> spatialPoints,
    std::vector<Mat> rotations, 
    std::vector<Mat> transitions, 
    Mat calibration
    );

/*
    Get euler angles by rotation matrix;
    Indexes:
    0 - bank (Z)
    1 - attitude (X) 
    2 - heading (Y)
    @param rotationMatrix rotation matrix

*/
Vec3f rotationMatrixToEulerAngles(
    Mat &rotationMatrix
    );
/*
    Keyboard events handler;
    @param w keyboard event
    @param t additional params for handler;

*/
void KeyboardViz3d(
    const viz::KeyboardEvent &w, 
    void *t
    );