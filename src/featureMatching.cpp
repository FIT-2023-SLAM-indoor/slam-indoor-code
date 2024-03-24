#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>

#include "featureMatching.h"

#include "config/config.h"
using namespace cv;

MatcherType getMatcherTypeIndex() {
	if (configService.getValue<bool>(ConfigFieldEnum::FM_SIFT_BF_))
		return MatcherType::SIFT_BF;
	if (configService.getValue<bool>(ConfigFieldEnum::FM_SIFT_FLANN_))
		return MatcherType::SIFT_FLANN;
	if (configService.getValue<bool>(ConfigFieldEnum::FM_ORB_))
		return MatcherType::ORB_BF;
	throw new std::exception();
}

void getMatchedPointCoords(
	std::vector<KeyPoint> &firstFeatures, std::vector<KeyPoint> &secondFeatures,
	std::vector<DMatch> &matches, std::vector<Point2f> &firstFrameMatchedPointCoords,
	std::vector<Point2f> &secondFrameMatchedPointCoords)
{
	firstFrameMatchedPointCoords.clear();
	secondFrameMatchedPointCoords.clear();
	for (int i = 0; i < matches.size(); i++) {
		firstFrameMatchedPointCoords.push_back(firstFeatures[matches[i].queryIdx].pt);
		secondFrameMatchedPointCoords.push_back(secondFeatures[matches[i].trainIdx].pt);
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

/**
 * Написать документацию!!!
*/
void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    int matcherType,
    std::vector<DMatch>& matches)
{
    // Extract descriptors from the key points of the input frames
    Mat firstDescriptor;
    extractDescriptor(firstFrame, firstFeatures, 
        matcherType, firstDescriptor);
    Mat secondDescriptor;
    extractDescriptor(secondFrame, secondFeatures, 
        matcherType, secondDescriptor);

    // Match the descriptors using the specified matcher type and radius
    matchFeatures(firstDescriptor, secondDescriptor, matches,
        matcherType);
}