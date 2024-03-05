#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
/*
* @param previousFeatures [in]
* @param currentFeatures [in]
* @param matches [in]
* @param matchedFeatures [out]
* @param newPreviousFeatures [out] 
*/
void getMatchedPoints(
	std::vector<KeyPoint>& previousFeatures,
	std::vector<KeyPoint>& currentFeatures,
	std::vector<DMatch> matches,
	std::vector<Point2f>& matchedFeatures,
	std::vector<Point2f>& newPreviousFeatures
);
/*
* @param prevDesc [in]
* @param curDesc [in]
* @param matches [out]
* @param matcherType [in] matcher type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf
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
* @param extractorType [in] extractor type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf
* @param desc [out]
*/
void extractDescriptor(
	Mat& frame,
	std::vector<KeyPoint>& features,
	int extractorType,
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

/**
 * Matches features between two frames.
 *
 * This function extracts descriptors from the key points of two input frames,
 * and then matches the descriptors using a specified matcher type and radius.
 *
 * @param [in] firstFrame The first input frame.
 * @param [in] secondFrame The second input frame.
 * @param [in] firstFeatures The key points of the first input frame.
 * @param [in] secondFeatures The key points of the second input frame.
 * @param [in] matcherType The type of descriptor matcher to use.
 * @param [in] radius The matching radius.
 * @param [out] matches The output vector of DMatch objects containing the matches.
 */
void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
	std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    int matcherType,
    float radius,
    std::vector<DMatch>& matches
);
