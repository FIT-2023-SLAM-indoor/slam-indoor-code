#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraCalibration.h"

#include "cameraTransition.h"

using namespace cv;

void countMatricies(InputArray qPoints, InputArray gPoints)
{
	Mat fundamentalMatrix = findFundamentalMat(qPoints, gPoints);

	// How to find K camera calibration matrix?
	Mat cameraCalib(3, 3, CV_32F);
    calibration(cameraCalib, CalibrationOption::load);

	Mat essentialMatrix = findEssentialMat(qPoints, gPoints, cameraCalib);

	// Find P (extrinsic matrix) using SVD class
	// Find rotations and translation using wrapped SVD
	Mat rot1, rot2, transition, negativeTransition;
	decomposeEssentialMat(essentialMatrix, rot1, rot2, transition);

    Mat possibleP1(3, 4, CV_32F),
        possibleP2(3, 4, CV_32F),
        possibleP3(3, 4, CV_32F),
        possibleP4(3, 4, CV_32F);
    hconcat(rot1, transition, possibleP1);
    hconcat(rot1, -transition, possibleP2);
    hconcat(rot2, transition, possibleP3);
    hconcat(rot2, -transition, possibleP4);

    std::cout << "P1: " << possibleP1 << std::endl;
    std::cout << "P1: " << possibleP2 << std::endl;
    std::cout << "P1: " << possibleP3 << std::endl;
    std::cout << "P1: " << possibleP4 << std::endl;

    // TODO: add choosing of correct P matrix using the first point
}
