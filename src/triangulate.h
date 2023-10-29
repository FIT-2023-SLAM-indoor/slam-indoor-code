#pragma once
#include "opencv2/core/core_c.h"

void triangulate(cv::InputArray projPoints1, cv::InputArray projPoints2,
    cv::Mat matr1, cv::Mat matr2,
    cv::OutputArray points4D);