#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "../../misc/IOmisc.h"
#include "../../misc/ChronoTimer.h"

#include "featureMatching.h"
#include "featureMatchingCommon.h"

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaimgproc.hpp"
#endif

using namespace cv;

#ifdef USE_CUDA
static void matchFeatures(
	cuda::GpuMat& prevDesc,
	cuda::GpuMat& curDesc,
	std::vector<DMatch>& matches,
	int extractorType
) {
	Ptr<cuda::DescriptorMatcher> matcher;
	switch (extractorType) {
		case SIFT_BF:
			matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L1);
			break;
		case SIFT_FLANN:
			matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
			break;
		case ORB_BF:
			matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
			break;
		default:
			throw std::exception();
	}

	cuda::GpuMat allMatchesGpu;
	matcher->knnMatchAsync(prevDesc, curDesc, allMatchesGpu, 2);

	std::vector<std::vector<DMatch>> allMatches;
	matcher->knnMatchConvert(allMatchesGpu, allMatches);
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
	ChronoTimer timer;
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
	ChronoTimer timer;
	Mat secondDescriptor;
	extractDescriptor(secondFrame, secondFeatures,
		matcherType, secondDescriptor);

	cuda::GpuMat firstDescriptorGpu(firstFrameDescriptor);
	cuda::GpuMat secondDescriptorGpu(secondDescriptor);

	timer.printLastPointDelta("Descriptors extracting: ", logStreams.timeStream);
	timer.updateLastPoint();

	matchFeatures(firstDescriptorGpu, secondDescriptorGpu, matches,
		matcherType);

	timer.printLastPointDelta("Matching: ", logStreams.timeStream);
}
#endif