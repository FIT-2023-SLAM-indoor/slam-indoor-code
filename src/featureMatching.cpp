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
	extractor = cv::SIFT::create();
#endif 
#ifdef FM_SIFT_FLANN 
	extractor = cv::SIFT::create();
#endif 
#ifdef FM_ORB
	extractor = cv::ORB::create();
#endif // FM_ORB
	extractor->compute(previousFrame, previousFeatures, prevImgDesc);
	extractor->compute(currentFrame, currentFeatures, curImgDesc);
#ifdef FM_SIFT_BF
	matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
#endif // FM_SIFT
#ifdef FM_SIFT_FLANN 
	matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
#endif 

#ifdef FM_ORB
	matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
#endif // FM_ORB
	matcher->knnMatch(prevImgDesc, curImgDesc, matches, 2);
	
	std::vector<DMatch> good_matches;
	const float ratio_thresh = 0.7f;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
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
	waitKey(10000);
#endif // SHOW_TRACKED_POINTS

}
