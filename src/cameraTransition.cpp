#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <random>


#include "cameraTransition.h"

using namespace cv;


void estimateProjection(cv::InputArray points1, cv::InputArray points2, const cv::Mat& calibrationMatrix,
                        cv::Mat& rotationMatrix, cv::Mat& translationVector, cv::Mat& projectionMatrix)
{

    // Maybe it's have sense to undistort points and matrix K

	Mat essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);
    std::cout  << "E:\n" << essentialMatrix << std::endl;

    // Find P matrix using wrapped SVD and
    int randomPointIndex = rand() % points1.rows();
    Mat p1(1, 2, CV_64F), p2(1, 2, CV_64F);
    points1.getMat().row(randomPointIndex).copyTo(p1.row(0));
    points2.getMat().row(randomPointIndex).copyTo(p2.row(0));
    std::cout << recoverPose(essentialMatrix, p1, p1, rotationMatrix, translationVector) << std::endl;
    hconcat(rotationMatrix, translationVector, projectionMatrix);
    std::cout << projectionMatrix << std:: endl;
}
