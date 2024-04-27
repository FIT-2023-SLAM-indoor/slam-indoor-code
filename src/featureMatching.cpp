#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "misc/ChronoTimer.h"

#include <opencv2/highgui.hpp>

#include "featureMatching.h"

#include "config/config.h"

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaimgproc.hpp"
#endif

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

void matchFeatures(
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

	std::cout << "START KNN MATCHING\n";

	matcher->knnMatch(prevDesc, curDesc, allMatches, 2);

	std::cout << "KNN MATCHED\n";

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

#ifdef USE_CUDA
void matchFeaturesCUDA(
	cuda::GpuMat& prevDesc,
	cuda::GpuMat& curDesc,
	std::vector<DMatch>& matches,
	int extractorType
) {
	Ptr<cuda::DescriptorMatcher> matcher;
	switch (extractorType) {
		case SIFT_BF:
			matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
			break;
		default:
			std::cerr << "Only SIFT bruteforce matcher is supported using CUDA" << std::endl;
			throw std::exception();
	}

	std::vector<std::vector<DMatch>> allMatches;
	matcher->knnMatch(prevDesc, curDesc, allMatches, 2);
	getGoodMatches(allMatches,matches);
}

void extractDescriptorCUDA(
	Mat& frame,
	std::vector<KeyPoint>& features,
	int matcherType,
	cuda::GpuMat& desc
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
	Mat descriptors;
	extractor->compute(frame, features, descriptors);
	desc.upload(descriptors);

	// WIP...
//	cv::Ptr<cuda::ORB> extractor = cuda::ORB::create();
//
//	std::vector<cv::Point2d> points;
//	for(auto &feature : features)
//	{
//		points.push_back(feature.pt);
//	}
//
//	cuda::GpuMat imageGpu(frame), keyPointsGpu(points);
//	cuda::GpuMat grayImage;
//	cuda::cvtColor(imageGpu, grayImage, COLOR_BGR2GRAY);
//
//	std::cout << keyPointsGpu.rows << " " << keyPointsGpu.cols << std::endl;
//
//	extractor->computeAsync(grayImage, keyPointsGpu, desc);
}

void matchFramesPairFeaturesCUDA(
	Mat& firstFrame,
	Mat& secondFrame,
	std::vector<KeyPoint>& firstFeatures,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
) {
	ChronoTimer timer;
	cuda::GpuMat firstDescriptor;
	extractDescriptorCUDA(firstFrame, firstFeatures,
		matcherType, firstDescriptor);
	matchFramesPairFeaturesCUDA(firstDescriptor, secondFrame, secondFeatures, matcherType, matches);
}

void matchFramesPairFeaturesCUDA(
	cuda::GpuMat& firstFrameDescriptor,
	Mat& secondFrame,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
) {
	ChronoTimer timer;
	cuda::GpuMat secondDescriptor;
	extractDescriptorCUDA(secondFrame, secondFeatures,
		matcherType, secondDescriptor);

	timer.printLastPointDelta("Descriptors extracting: ", std::cout);
	timer.updateLastPoint();

	matchFeaturesCUDA(firstFrameDescriptor, secondDescriptor, matches,
		matcherType);

	timer.printLastPointDelta("Matching: ", std::cout);
}
#endif