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
	SVD svdSolver(essentialMatrix, SVD::NO_UV);
	Mat extrinsicMatrix = svdSolver.w; // or vt

    std::cout << "P from SVD: " << extrinsicMatrix << std::endl;

	// Find rotations and translation using wrapped SVD
	Mat rot1, rot2, translation;
	decomposeEssentialMat(essentialMatrix, rot1, rot2, translation);

    std::cout << "R1: " << rot1 << std::endl;
    std::cout << "R2: " << rot2 << std::endl;
    std::cout << "t: " << translation << std::endl;

    Mat extendedR1(3, 4, CV_32F), extendedR2(3, 4, CV_32F);
    hconcat(rot1, translation, extendedR1);
    hconcat(rot2, translation, extendedR2);

    std::cout << "R1: " << extendedR1 << std::endl;
    std::cout << "R2: " << extendedR2 << std::endl;

    Mat P1(3,4,CV_32F), P2(3, 4, CV_32F);
    Mat D;
    D.diag(4);
    gemm(cameraCalib, extendedR1, 1, D, 1, P1); // TODO: Понять, как их нормально перемножать
    std::cout << "P1: " << P1 << std::endl;
    std::cout << "P2: " << P2 << std::endl;
}
