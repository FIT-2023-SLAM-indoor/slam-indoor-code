#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
/*
* @param previousFeatures [in]
* @param currentFeatures [in]
* @param matches [in]
* @param matchedFeatures [out]
* @param newPreviousFeatures [out]
* @param matcherType [in]
* @param radius [in]
*/
void getMatchedPoints(
	std::vector<KeyPoint>& previousFeatures,
	std::vector<KeyPoint>& currentFeatures,
	std::vector<DMatch> matches,
	std::vector<Point2f>& matchedFeatures,
	std::vector<Point2f>& newPreviousFeatures,
	int matcherType,
	float radius
);
/*
* @param prevDesc [in]
* @param curDesc [in]
* @param matches [out]
* @param matcherType [in]
* @param radius [in]
*/
void matchFeatures(
	Mat& prevDesc,
	Mat& curDesc,
	std::vector<DMatch>& matches,
	int matcherType,
	float radius
);
/*
* @param allMatches [in]
* @param matches [out]
*/
void getGoodMatches(
	std::vector<std::vector<DMatch>>& allMatches,
	std::vector<DMatch>& matches
);
/*
* @param frame [in]
* @param features [in]
* @param matcherType [in]
* @param desc [out]
*/
void extractDescriptor(
	Mat& frame,
	std::vector<KeyPoint>& features,
	int matcherType,
	Mat& desc
);
/*
* @param previousFeatures [in]
* @param currentFeatures [in]
* @param previousFrame [in]
* @param currentFrame [in]
* @param matches [in]
*/
void showMatchedPointsInTwoFrames(
	std::vector<KeyPoint>& previousFeatures,
	std::vector<KeyPoint>& currentFeatures,
	Mat& previousFrame,
	Mat& currentFrame,
	std::vector<DMatch>& matches
);



