#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#pragma once
/**
*
*
*
*/
int videoProcessingCycle(cv::VideoCapture& cap, int featureTrackingBarier, int featureTrackingMaxAcceptableDiff,
	int framesGap, int requiredExtractedPointsCount, int featureExtractingThreshold);