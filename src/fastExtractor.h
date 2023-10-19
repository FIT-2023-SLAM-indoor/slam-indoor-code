#pragma once
#include <opencv2/opencv.hpp>

void fastExtractor(cv::Mat& srcImage, std::vector<cv::KeyPoint>& points,
    int threshold = 10, bool suppression = true, 
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16);