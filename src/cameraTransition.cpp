#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraTransition.h"

using namespace cv;

#define DISTANCE_THRESHOLD 100

bool estimateProjection(InputArray points1, InputArray points2, const Mat& calibrationMatrix,
	Mat& rotationMatrix, Mat& translationVector, Mat& projectionMatrix, Mat& triangulatedPoints)
{

	// Maybe it's have sense to undistort points and matrix K

	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);

    // Choose one random corresponding points pair
    int randomPointIndex = rand() % points1.rows();
    Mat p1(1, 2, CV_64F), p2(1, 2, CV_64F);
    points1.getMat().row(randomPointIndex).copyTo(p1.row(0));
    points2.getMat().row(randomPointIndex).copyTo(p2.row(0));

	// Find P matrix using wrapped OpenCV SVD and triangulation
	int passedPointsCount = recoverPose(essentialMatrix, points1, points2, calibrationMatrix,
                                        rotationMatrix, translationVector,
                                        DISTANCE_THRESHOLD, noArray(), triangulatedPoints);
	hconcat(rotationMatrix, translationVector, projectionMatrix);

    // Here bundle adjustment can be implemented

	return passedPointsCount > 0;
}

void refineWorldCameraPose(Mat& rotationMatrix, Mat& translationVector,
                           Mat& worldCameraPose, Mat& worldCameraRotation)
{
//    std::cout << (rotationMatrix.type() == CV_64F) << " " << (worldCameraRotation.type() == CV_32F) << std::endl;
    worldCameraRotation *= rotationMatrix;
    worldCameraPose += worldCameraRotation * translationVector;
}

void addHomogeneousRow(Mat& m) {
    Mat row = Mat::zeros(1, m.cols, CV_64F);
    row.at<double>(0, m.cols-1) = 1;
    m.push_back(row);
}

void removeHomogeneousRow(Mat& m) {
    m.pop_back();
}

void combineProjectionMatrices(cv::Mat& p1, cv::Mat& p2, cv::Mat& res) {
    Mat lastRow = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    Mat p1H = p1.clone(), p2H = p2.clone();
    p1H.push_back(lastRow.clone());
    p2H.push_back(lastRow.clone());
    res = p1H * p2H;
}