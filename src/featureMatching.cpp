#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

#include "featureMatching.h"

#include "config/config.h"
using namespace cv;

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

void matchFeatures(
	Mat& prevDesc, 
	Mat& curDesc, 
	std::vector<DMatch>& matches,
	int extractorType
) {
	std::vector<std::vector<DMatch>> allMatches;
	cv::Ptr<cv::DescriptorExtractor> extractor;

	Ptr<DescriptorMatcher> matcher;
	switch (extractorType) {
	case SIFT_BF:
		matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
		break;
	case SIFT_FLANN:
		matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		break;
	case ORB_BF:
		matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
		break;
	default:
		throw std::exception();
	}
	matcher->knnMatch(prevDesc, curDesc, allMatches, 2);
	getGoodMatches(allMatches,matches);
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

void extractDescriptor(
	Mat& frame,
	std::vector<KeyPoint>& features,
	int matcherType,
	Mat& desc
)
{
	cv::Ptr<cv::DescriptorExtractor> extractor;
	switch (matcherType) {
	case SIFT_BF:
		extractor = cv::SIFT::create();
		break;
	case SIFT_FLANN:
		extractor = cv::SIFT::create();
		break;
	case ORB_BF:
		extractor = cv::ORB::create();
		break;
	default:
		throw std::exception();
	}
	extractor->compute(frame, features, desc);
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

void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    int matcherType,
    std::vector<DMatch>& matches
) {
    Mat firstDescriptor;
    extractDescriptor(firstFrame, firstFeatures,
        matcherType, firstDescriptor);
	matchFramesPairFeatures(firstDescriptor, secondFrame, secondFeatures, matcherType, matches);
}

void matchFramesPairFeatures(
	Mat& firstFrameDescriptor,
	Mat& secondFrame,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
) {
	Mat secondDescriptor;
	extractDescriptor(secondFrame, secondFeatures,
		matcherType, secondDescriptor);

	matchFeatures(firstFrameDescriptor, secondDescriptor, matches,
		matcherType);
}