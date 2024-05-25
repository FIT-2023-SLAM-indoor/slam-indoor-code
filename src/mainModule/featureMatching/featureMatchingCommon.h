#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
/**
* Module with common functions for CUDA and CPU matching
*/

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
 * точек. Линейным обходом вектора matches функция вычисляет координаты (x, y) точек первого и
 * второго кадра из соответсвующего матча. Проще говоря мы получаем для каждой ключевой точки её
 * координаты на первом и втором изображении.
 *
 * @param [in] prevFrameFeatures
 * @param [in] nextFrameFeatures
 * @param [in] matches
 * @param [out] keyPointFrameCoords1
 * @param [out] keyPointFrameCoords2
 */
void getKeyPointCoordsFromFramePair(
	const std::vector<KeyPoint> &prevFrameFeatures,
	const std::vector<KeyPoint> &nextFrameFeatures,
	const std::vector<DMatch> &matches,
	std::vector<Point2f> &keyPointFrameCoords1,
	std::vector<Point2f> &keyPointFrameCoords2
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