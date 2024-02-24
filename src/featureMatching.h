#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
/**
* Feature matching function.
*/
void featureMatching(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& trackedFeatures, std::vector<Point2f>& newPreviousFeatures);