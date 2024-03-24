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

void getMatchedPoints(
	std::vector<KeyPoint>& previousFeatures,
	std::vector<KeyPoint>& currentFeatures,
	std::vector<DMatch> matches,
	std::vector<Point2f>& matchedFeatures,
	std::vector<Point2f>& newPreviousFeatures
) {
	matchedFeatures.clear();
	newPreviousFeatures.clear();
	for (int i = 0;i < matches.size();i++) {
		matchedFeatures.push_back(currentFeatures.at(
			matches[i].trainIdx).pt);
		newPreviousFeatures.push_back(previousFeatures.at(
			matches[i].queryIdx).pt);
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