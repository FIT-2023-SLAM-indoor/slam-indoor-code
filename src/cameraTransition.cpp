#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraTransition.h"
#include "main_config.h"

using namespace cv;

static void filterVectorByMask(std::vector<Point2f>& oldVector, const Mat& mask) {
//    Mat mask;
//    inputMask.convertTo(mask, CV_8U);
    std::vector<Point2f> newVector;
    for (int i = 0; i < mask.rows; ++i) {
            if (mask.at<uchar>(i))
                newVector.push_back(oldVector[i]);
    }
    oldVector = newVector;
}

bool estimateProjection(std::vector<Point2f>& points1, std::vector<Point2f>& points2, const Mat& calibrationMatrix,
	Mat& rotationMatrix, Mat& translationVector, Mat& projectionMatrix, Mat& triangulatedPoints)
{

	// Maybe it's have sense to undistort points and matrix K
    double focal_length = 0.5*(calibrationMatrix.at<double>(0) + calibrationMatrix.at<double>(4));
    Point2d principle_point(calibrationMatrix.at<double>(2), calibrationMatrix.at<double>(5));
    Mat mask;
#ifdef USE_RANSAC
	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix, RANSAC,
                                           RANSAC_PROB, RANSAC_THRESHOLD, mask);
#else
    Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);
#endif
    if (essentialMatrix.empty())
        return false;

#ifdef USE_RANSAC
    double maskNonZeroElemsCnt = countNonZero(mask);
    std::cout << "Used in RANSAC E matrix estimation: " << maskNonZeroElemsCnt << std::endl;
    filterVectorByMask(points1, mask);
    filterVectorByMask(points2, mask);
//    if ((maskNonZeroElemsCnt / points1.size()) < RANSAC_GOOD_POINTS_PERCENT)
//        return false;
#endif

	// Find P matrix using wrapped OpenCV SVD and triangulation
    Mat recoverMask;
	int passedPointsCount = recoverPose(essentialMatrix, points1, points2, calibrationMatrix,
                                        rotationMatrix, translationVector,
                                        RECOVER_POSE_DISTANCE_THRESHOLD, recoverMask, triangulatedPoints);
	hconcat(rotationMatrix, translationVector, projectionMatrix);
    maskNonZeroElemsCnt = countNonZero(recoverMask);
    std::cout << "Passed cherality check cnt: " << maskNonZeroElemsCnt << std::endl;
//    filterVectorByMask(points1, recoverMask);
//    filterVectorByMask(points2, recoverMask);
	return passedPointsCount > 0;
}

void refineWorldCameraPose(Mat& rotationMatrix, Mat& translationVector,
                           Mat& worldCameraPose, Mat& worldCameraRotation)
{
//    std::cout << (rotationMatrix.type() == CV_64F) << " " << (worldCameraRotation.type() == CV_32F) << std::endl;
    worldCameraRotation = rotationMatrix * worldCameraRotation;
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
