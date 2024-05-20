#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "../misc/ChronoTimer.h"

#include "featureMatching.h"
#include "featureMatchingCommon.h"

using namespace cv;

/*
* @param prevDesc [in]
* @param curDesc [in]
* @param matches [out]
* @param matcherType [in] matcher type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf
*/
static void matchFeatures(
	Mat& prevDesc, 
	Mat& curDesc, 
	std::vector<DMatch>& matches,
	int extractorType
) {
	std::vector<std::vector<DMatch>> allMatches;

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

void extractDescriptor(
	Mat& frame,
	std::vector<KeyPoint>& features,
	int matcherType,
	Mat& desc
) {
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

void matchFramesPairFeatures(
	Mat& firstFrame,
	Mat& secondFrame,
	std::vector<KeyPoint>& firstFeatures,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
) {
	// Extract descriptors from the key points of the input frames
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