#include <opencv2/opencv.hpp>

#include "fastExtractor.h"

/*
 * Detects corners using the FAST algorithm and writes info about them into vector.
 * 
 * @param image where keypoints (corners) are detected.
 * 
 * @param points keypoints detected on the image.
 * 
 * @param threshold on difference between intensity of 
 *     the central pixel and pixels of a circle around this pixel. 
 *
 * @param suppression if true, non-maximum suppression 
 *     is applied to detected corners (keypoints).
 *
 * @param type one of the three neighborhoods as defined in the paper.
 */
void fastExtractor(cv::Mat& srcImage, std::vector<cv::KeyPoint>& points,
    int threshold, bool suppression, cv::FastFeatureDetector::DetectorType type)
{
    // Set the key point detector settings.
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(threshold, suppression, type);

    // Detects keypoints in srcImage. @param points is the detected features.
    detector->detect(srcImage, points);
}