#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "poseEstimation.h"

void countMatricies(cv::InputArray qPoints, cv::InputArray gPoints)
{
	cv::Mat fundamentalMatrix = cv::findFundamentalMat(qPoints, gPoints);

	// How to find K camera calibration matrix?
	cv::Mat cameraCalib(3, 3, CV_32F);
	// cv::calibrateCamera();

	cv::Mat essentialMatrix = cv::findEssentialMat(qPoints, gPoints, cameraCalib);

	// Find P (extrnisic matrix) using SVD class
	cv::SVD svdSoler(essentialMatrix);
	cv::Mat extrinsicMatrix = svdSoler.w;

	// Find rotations and translation using wrapped SVD
	cv::Mat rot1, rot2, translation;
	cv::decomposeEssentialMat(essentialMatrix, rot1, rot2, translation);
}
