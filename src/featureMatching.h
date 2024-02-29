#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
/**
* Feature matching function. Use defines FM_SIFT_BF FM_ORB and FM_SIFT_FLANN to select your FM algo.
*/
void featureMatching(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& trackedFeatures, std::vector<Point2f>& newPreviousFeatures);