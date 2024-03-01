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
/*
* Main function for feature matching, matcher type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf.
*/
void getMatchedPoints(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& matchedFeatures, std::vector<Point2f>& newPreviousFeatures, int matcherType, int radius, bool showMatchedPoints) {

	Mat prevDesc, curDesc;
	std::vector<std::vector<DMatch>> matches;
	extractDescriptorsAndMatch(previousFrame, currentFrame, previousFeatures, currentFeatures, 
		prevDesc, curDesc, matcherType, matches, radius);
	getGoodMatches(previousFeatures, currentFeatures, matchedFeatures, newPreviousFeatures, matches);
	if (showMatchedPoints) {
		showMatchedPointsInTwoFrames(previousFeatures, currentFeatures,
			previousFrame, currentFrame, matches);
	}

}

void getGoodMatches(std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& matchedFeatures, std::vector<Point2f>& newPreviousFeatures, std::vector<std::vector<DMatch>>& matches) {
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].size() > 0)
		{
			matchedFeatures.push_back(currentFeatures.at(
				matches[i][0].trainIdx).pt);
			newPreviousFeatures.push_back(previousFeatures.at(
				matches[i][0].queryIdx).pt);
		}
	}
}

void extractDescriptorsAndMatch(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	Mat& prevImgDesc, Mat& curImgDesc,int matcherType, std::vector<std::vector<DMatch>>& matches, int radius) {
	cv::Ptr<cv::DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;
	switch (matcherType) {
		case SIFT_BF:
			matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
			extractor = cv::SIFT::create();
			break;
		case SIFT_FLANN:
			matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
			extractor = cv::SIFT::create();
			break;
		case ORB_BF:
			extractor = cv::ORB::create();
			matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
			break;
		default:
			throw std::exception("No such type");
	}
	extractor->compute(previousFrame, previousFeatures, prevImgDesc);
	extractor->compute(currentFrame, currentFeatures, curImgDesc);
	matcher->radiusMatch(prevImgDesc, curImgDesc, matches, radius);
}

void showMatchedPointsInTwoFrames(std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	Mat& previousFrame, Mat& currentFrame, std::vector<std::vector<DMatch>>& matches) {
	Mat output_image;
	std::vector<DMatch> goodMatches;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].size() > 0)
		{
			goodMatches.push_back(matches[i].at(0));
		}
	}
	cv::drawMatches(
		previousFrame, previousFeatures,
		currentFrame, currentFeatures,
		goodMatches,
		output_image, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("matches", output_image);
	waitKey(3000);
	goodMatches.clear();
}
