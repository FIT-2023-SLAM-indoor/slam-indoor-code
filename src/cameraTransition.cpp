#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraTransition.h"
#include "main_config.h"

using namespace cv;

void filterVectorByMask(std::vector<Point2f>& vector, const Mat& mask) {
    std::vector<Point2f> newVector;
	Mat filterMask = mask;
	if (filterMask.rows == 1 && filterMask.cols != 1)
		filterMask = filterMask.t();
    for (int i = 0; i < filterMask.rows; ++i) {
            if (filterMask.at<uchar>(i))
                newVector.push_back(vector[i]);
    }
	vector = newVector;
}

bool estimateTransformation(
		const std::vector<Point2f>& points1, const std::vector<Point2f>& points2, const Mat& calibrationMatrix,
		Mat& rotationMatrix, Mat& translationVector, Mat& chiralityMask
) {
	Mat mask;
	double maskNonZeroElemsCnt = 0;
#ifdef USE_RANSAC
	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix, RANSAC,
										   RANSAC_PROB, RANSAC_THRESHOLD, mask);
#else
	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);
#endif
	if (essentialMatrix.empty())
		return false;

#ifdef USE_RANSAC
	maskNonZeroElemsCnt = countNonZero(mask);
	std::cout << "Used in RANSAC E matrix estimation: " << maskNonZeroElemsCnt << std::endl;
//    if ((maskNonZeroElemsCnt / points1.size()) < RANSAC_GOOD_POINTS_PERCENT)
//        return false;
#endif

	// Find P matrix using wrapped OpenCV SVD and triangulation
	int passedPointsCount = recoverPose(essentialMatrix, points1, points2, calibrationMatrix,
										rotationMatrix, translationVector,
										RECOVER_POSE_DISTANCE_THRESHOLD, chiralityMask);
	maskNonZeroElemsCnt = countNonZero(chiralityMask);
	std::cout << "Passed chirality check cnt: " << maskNonZeroElemsCnt << std::endl;
	return passedPointsCount > 0;
}

bool estimateProjection(std::vector<Point2f>& points1, std::vector<Point2f>& points2, const Mat& calibrationMatrix,
	Mat& rotationMatrix, Mat& translationVector, Mat& projectionMatrix, Mat& triangulatedPoints)
{

    Mat mask;
	double maskNonZeroElemsCnt = 0;
#ifdef USE_RANSAC
	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix, RANSAC,
                                           RANSAC_PROB, RANSAC_THRESHOLD, mask);
#else
    Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);
#endif
    if (essentialMatrix.empty())
        return false;

#ifdef USE_RANSAC
    maskNonZeroElemsCnt = countNonZero(mask);
    std::cout << "Used in RANSAC E matrix estimation: " << maskNonZeroElemsCnt << std::endl;
#ifdef USE_RANSAC_POINTS_FILTER
    filterVectorByMask(points1, mask);
    filterVectorByMask(points2, mask);
#endif
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
#ifdef USE_RECOVER_POSE_POINTS_FILTER
    filterVectorByMask(points1, recoverMask);
    filterVectorByMask(points2, recoverMask);
#endif
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
