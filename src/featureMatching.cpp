#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "featureMatching.h"

#include "main_config.h"
using namespace cv;
void featureMatching(Mat& previousFrame, Mat& currentFrame, std::vector<KeyPoint>& previousFeatures, std::vector<KeyPoint>& currentFeatures,
	std::vector<Point2f>& trackedFeatures, std::vector<Point2f>& newPreviousFeatures) {
	
	Mat prevImgDesc, curImgDesc;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	std::vector<std::vector<DMatch>> matches;
	Ptr<DescriptorMatcher> matcher;
#ifdef FM_SIFT_BF 
	const float ratio_thresh = 0.6f;
	extractor = cv::SIFT::create();
	extractor->compute(previousFrame, previousFeatures, prevImgDesc);
	extractor->compute(currentFrame, currentFeatures, curImgDesc);
	matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
#elif defined  FM_SIFT_FLANN 
	const float ratio_thresh = 0.5f;
	extractor = cv::SIFT::create();
	extractor->compute(previousFrame, previousFeatures, prevImgDesc);
	extractor->compute(currentFrame, currentFeatures, curImgDesc);
	matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
#elif defined FM_ORB
	const float ratio_thresh = 0.7f;
	extractor = cv::ORB::create();
	extractor->compute(previousFrame, previousFeatures, prevImgDesc);
	extractor->compute(currentFrame, currentFeatures, curImgDesc);
	matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
#else
	const float ratio_thresh = 0.5f;
	throw std::exception("MM is not chosen");
#endif // FM_ORB
	matcher->radiusMatch(prevImgDesc, curImgDesc, matches, FM_SEARCH_RADIUS);
	std::vector<DMatch> good_matches;
	
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i].size() > 0)
		{
			good_matches.push_back(matches[i][0]);
			//std::cout << currentFrameExtractedKeyPoints.size() << " 1 " << matches[i][0].trainIdx << std::endl;
			//std::cout << previousFrameExtractedKeyPoints.size() << " 2 " << matches[i][0].queryIdx << std::endl;
			trackedFeatures.push_back(currentFeatures.at(
				matches[i][0].trainIdx).pt);
			newPreviousFeatures.push_back(previousFeatures.at(
				matches[i][0].queryIdx).pt);
		}
	}

#ifdef SHOW_TRACKED_POINTS1
	Mat output_image;
	cv::drawMatches(
		previousFrame, previousFeatures,
		currentFrame, currentFeatures,
		good_matches,
		output_image, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("ddd", output_image);
	waitKey(3000);
#endif // SHOW_TRACKED_POINTS

	previousFeatures.clear();
	currentFeatures.clear();
;}
