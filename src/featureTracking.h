#pragma once
#include <opencv2/opencv.hpp>

void getPointsAroundFeature(cv::Point2f feature, int radius, double barier, std::vector<cv::Point2f>& pointsAround);

double sumSquaredDifferences(std::vector<cv::Point2f>& batch1, std::vector<cv::Point2f>& batch2, cv::Mat& image1, cv::Mat& image2, double min);

int trackFeature(cv::Point2f feature, cv::Mat& image1, cv::Mat& image2, cv::Point2f& res, double barier);

void trackFeatures(std::vector<cv::Point2f>& features, cv::Mat& previousFrame, cv::Mat& currentFrame, std::vector<cv::Point2f>& newFeatures,
	int barier, double maxAcceptableDifference);