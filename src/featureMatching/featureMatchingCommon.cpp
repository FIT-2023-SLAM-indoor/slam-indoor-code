#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "../misc/ChronoTimer.h"

#include <opencv2/highgui.hpp>

#include "../config/config.h"

#include "featureMatchingCommon.h"


MatcherType getMatcherTypeIndex() {
	if (configService.getValue<bool>(ConfigFieldEnum::FM_SIFT_BF))
		return MatcherType::SIFT_BF;
	if (configService.getValue<bool>(ConfigFieldEnum::FM_SIFT_FLANN))
		return MatcherType::SIFT_FLANN;
	if (configService.getValue<bool>(ConfigFieldEnum::FM_ORB))
		return MatcherType::ORB_BF;
	throw new std::exception();
}

void getKeyPointCoordsFromFramePair(const std::vector<KeyPoint> &prevFrameFeatures,
									const std::vector<KeyPoint> &nextFrameFeatures, const std::vector<DMatch> &matches,
									std::vector<Point2f> &keyPointFrameCoords1, std::vector<Point2f> &keyPointFrameCoords2)
{
	keyPointFrameCoords1.clear();
	keyPointFrameCoords2.clear();
	for (int i = 0; i < matches.size(); i++) {
		keyPointFrameCoords1.push_back(prevFrameFeatures[matches[i].queryIdx].pt);
		keyPointFrameCoords2.push_back(nextFrameFeatures[matches[i].trainIdx].pt);
	}
}



void getGoodMatches(
	std::vector<std::vector<DMatch>>& allMatches,
	std::vector<DMatch>& matches
){
	matches.clear();
	double distanceMlt = configService.getValue<double>(ConfigFieldEnum::FM_KNN_DISTANCE);
	for (size_t i = 0; i < allMatches.size(); i++)
	{
		if (allMatches[i].empty())
			continue;
		if (allMatches[i][0].distance < distanceMlt*allMatches[i][1].distance)
			matches.push_back(allMatches[i][0]);
	}
}

void showMatchedPointsInTwoFrames(
	std::vector<KeyPoint>& previousFeatures,
	std::vector<KeyPoint>& currentFeatures,
	Mat& previousFrame,
	Mat& currentFrame,
	std::vector<DMatch>& matches
) {
	Mat output_image;
	cv::drawMatches(
		previousFrame, previousFeatures,
		currentFrame, currentFeatures,
		matches,
		output_image, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("matches", output_image);
	waitKey(3000);
}