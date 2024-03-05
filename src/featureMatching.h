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
* @param previousFrame [in]
* @param currentFrame [in]
* @param previousFeatures [in]
* @param currentFeatures [in]
* @param matcherType [in]
* @param prevImgDesc [out]
* @param curImgDesc [out]
*/
void extractDescriptors(
	Mat& previousFrame,
	Mat& currentFrame,
	std::vector<KeyPoint>& previousFeatures,
	std::vector<KeyPoint>& currentFeatures,
	int matcherType,
	Mat& prevImgDesc,
	Mat& curImgDesc
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



