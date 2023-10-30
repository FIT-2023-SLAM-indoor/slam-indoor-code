#include <opencv2/opencv.hpp>

#include "fastExtractor.h"


void fastExtractor(cv::Mat& srcImage, std::vector<cv::KeyPoint>& points,
    int threshold, bool suppression, cv::FastFeatureDetector::DetectorType type)
{
    // Set the key point detector settings.
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(threshold, suppression, type);

    // Detects keypoints in srcImage. @param points is the detected features.
    detector->detect(srcImage, points);
}