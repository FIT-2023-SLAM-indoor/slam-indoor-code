#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraTransition.h"

using namespace cv;

#define DISTANCE_THRESHOLD 100

bool estimateProjection(std::vector<Point2f>& points1, std::vector<Point2f>& points2, const Mat& calibrationMatrix,
	Mat& rotationMatrix, Mat& translationVector, Mat& projectionMatrix, Mat& triangulatedPoints)
{

	// Maybe it's have sense to undistort points and matrix K
    double focal_length = 0.5*(calibrationMatrix.at<double>(0) + calibrationMatrix.at<double>(4));
    Point2d principle_point(calibrationMatrix.at<double>(2), calibrationMatrix.at<double>(5));
    Mat mask;
//	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix, RANSAC, 0.999, 1, mask);
    Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);
    if (essentialMatrix.empty())
        return false;

//    double maskNonZeroElemsCnt = countNonZero(mask);
//    std::cout << maskNonZeroElemsCnt << std::endl;
//    if ((maskNonZeroElemsCnt / points1.size()) < 0.6)
//        return false;

	// Find P matrix using wrapped OpenCV SVD and triangulation
	int passedPointsCount = recoverPose(essentialMatrix, points1, points2, calibrationMatrix,
                                        rotationMatrix, translationVector,
                                        DISTANCE_THRESHOLD, noArray(), triangulatedPoints);
	hconcat(rotationMatrix, translationVector, projectionMatrix);

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
