#include <opencv2/opencv.hpp>

#include "fastExtractor.h"

using namespace cv;

void fastExtractor(cv::Mat& srcImage, std::vector<cv::KeyPoint>& points,
    int threshold, bool suppression, cv::FastFeatureDetector::DetectorType type)
{
    Ptr<FastFeatureDetector> detector =
        FastFeatureDetector::create(threshold, suppression, type);
    detector->detect(srcImage, points);

}