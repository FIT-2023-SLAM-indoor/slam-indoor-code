#pragma once
#include <opencv2/opencv.hpp>

/*
 * Detects corners using the FAST algorithm and writes info about them into vector.
 *
 * @param srcImage where keypoints (corners) are detected.
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
    int threshold = 10, bool suppression = true, 
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16);