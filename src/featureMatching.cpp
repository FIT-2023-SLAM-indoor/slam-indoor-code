#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "featureMatching.h"

#include "main_config.h"
using namespace cv;
const int SIFT_BF = 0;
const int SIFT_FLANN = 1;
const int ORB_BF = 2;

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
	int extractorType,
	float radius
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
	matcher->radiusMatch(prevDesc, curDesc, allMatches, radius);
	getGoodMatches(allMatches,matches);

}

void getGoodMatches(
	std::vector<std::vector<DMatch>>& allMatches,
	std::vector<DMatch>& matches
){

	for (size_t i = 0; i < allMatches.size(); i++)
	{
		if (allMatches[i].size() > 0)
		{
			matches.push_back(allMatches[i][0]);
		}
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
