#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

enum MatcherType {
	SIFT_BF,
	SIFT_FLANN,
	ORB_BF
};
/**
 * Defines matcher type using flag from configService
 *
 * @return type as enum value
 */
MatcherType getMatcherTypeIndex();

/**
 * Функция принимает вектора ключевых точек первого и второго кадра, а также вектор матчей этих
 * точек. Линейным обходом вектора matches функция возвращает координаты (x, y) точек первого и
 * второго кадра из соответсвующего матча.
 *
 * @param [in] firstFrameFeatures
 * @param [in] secondFrameFeatures
 * @param [in] matches
 * @param [out] firstFrameMatchedPointCoords
 * @param [out] secondFrameMatchedPointCoords
 */
void getMatchedPointCoords(
	const std::vector<KeyPoint> &firstFrameFeatures, 
	const std::vector<KeyPoint> &secondFrameFeatures,
	const std::vector<DMatch> &matches, 
	std::vector<Point2f> &firstFrameMatchedPointCoords,
	std::vector<Point2f> &secondFrameMatchedPointCoords
);


/*
* @param prevDesc [in]
* @param curDesc [in]
* @param matches [out]
* @param matcherType [in] matcher type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf
*/
void matchFeatures(
	Mat& prevDesc,
	Mat& curDesc,
	std::vector<DMatch>& matches,
	int matcherType
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
 * Написать документацию!!!
*/
void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    int matcherType,
    std::vector<DMatch>& matches
);