#pragma once
#include <opencv2/calib3d.hpp>

void countMatrices(cv::InputArray qPoints, cv::InputArray gPoints, Mat& P);