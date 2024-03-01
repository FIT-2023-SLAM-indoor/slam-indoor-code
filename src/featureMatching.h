#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

void getMatchedPoints(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& matchedFeatures, std::vector<Point2f>& newPreviousFeatures, int matcherType, int radius, bool showMatchedPoints);

void showMatchedPointsInTwoFrames(std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	Mat& previousFrame, Mat& currentFrame, std::vector<std::vector<DMatch>>& matches);

void extractDescriptorsAndMatch(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	Mat& prevImgDesc, Mat& curImgDesc, int matcherType, std::vector<std::vector<DMatch>>& matches, int radius);

void getGoodMatches(std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& matchedFeatures, std::vector<Point2f>& newPreviousFeatures, std::vector<std::vector<DMatch>>& matches);