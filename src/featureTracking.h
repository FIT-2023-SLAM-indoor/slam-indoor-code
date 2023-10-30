#pragma once
#include <opencv2/opencv.hpp>
/*
 * Gets the feature and returns a std::vector of batch with given radius consists of points around this feature.
 * Barrier is used to choose the density of resulting circle;
 */
void getPointsAroundFeature(cv::Point2f feature, int radius, double barier, std::vector<cv::Point2f>& pointsAround);
/*
 * Gets two circle batches from two different pictures and returs sum of squared difference of pixels colors of given batches.
 */
double sumSquaredDifferences(std::vector<cv::Point2f>& batch1, std::vector<cv::Point2f>& batch2, cv::Mat& image1, cv::Mat& image2, double min);
/*
 * Tracking one feature.Gets feature,two images and address of feature in the second image to put the result in int.
 * @param barier is required for choosing density of circles.
 *
 */
int trackFeature(cv::Point2f feature, cv::Mat& image1, cv::Mat& image2, cv::Point2f& res, double barier);
/*
 * Tracking all the features.
 * @param features vector of previous features.
 * @param newFeatures vector of tracked features
 * @param barier is required to choose density of circles.
 *
 */
void trackFeatures(std::vector<cv::Point2f>& features, cv::Mat& previousFrame, cv::Mat& currentFrame, std::vector<cv::Point2f>& newFeatures,
	int barier, double maxAcceptableDifference);