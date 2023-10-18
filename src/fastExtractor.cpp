#include <opencv2/opencv.hpp>

#include "fastExtractor.h"

/*
 * 
 */
void fastExtractor(cv::Mat* srcImage, std::vector<cv::KeyPoint>* points,
    int threshold, bool suppression, cv::FastFeatureDetector::DetectorType type)
{
    //
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(threshold, suppression, type);

    //
    detector->detect(*srcImage, *points);
}