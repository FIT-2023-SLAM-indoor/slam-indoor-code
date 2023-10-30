#pragma once
#include <opencv2/calib3d.hpp>

void estimateIgorAss(cv::InputArray qPoints, cv::InputArray gPoints, Mat& P);