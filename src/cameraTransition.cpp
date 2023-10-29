#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraCalibration.h"

#include "cameraTransition.h"

using namespace cv;



void countMatrices(InputArray qPoints, InputArray gPoints, Mat& P)
{
	Mat fundamentalMatrix = findFundamentalMat(qPoints, gPoints);

	// How to find K camera calibration matrix?
	Mat cameraCalib(3, 3, CV_32F);
    calibration(cameraCalib, CalibrationOption::load);

	Mat essentialMatrix = findEssentialMat(qPoints, gPoints, cameraCalib);

    // Find P matrix using wrapped SVD and
    Mat cvR, cvt;
    Mat p1(1, 2, CV_32F), p2(1, 2, CV_32F);
    qPoints.getMat().row(0).copyTo(p1.row(0));
    gPoints.getMat().row(0).copyTo(p2.row(0));
    recoverPose(essentialMatrix, p1, p2, cvR, cvt);
    hconcat(cvR, cvt, P);


}
