#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;


/*
* @param prevDesc [in]
* @param curDesc [in]
* @param matches [out]
* @param matcherType [in] matcher type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf
*/
void matchFeatures(
	Mat& prevDesc,
	Mat& curDesc,
	std::vector<DMatch>& matches,
	int matcherType
);

/*
* @param frame [in]
* @param features [in]
* @param extractorType [in] extractor type: 0 - sift_bf, 1 - sift_flann, 2 - orb_bf
* @param desc [out]
*/
void extractDescriptor(
	Mat& frame,
	std::vector<KeyPoint>& features,
	int extractorType,
	Mat& desc
);

/**
 * For common case.
 *
 * @param firstFrame
 * @param secondFrame
 * @param firstFeatures
 * @param secondFeatures
 * @param matcherType
 * @param matches
 */
void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    int matcherType,
    std::vector<DMatch>& matches
);

/**
 * For case when the first descriptor is pre-calculated.
 *
 * @param firstFrameDescriptor
 * @param secondFrame
 * @param secondFeatures
 * @param matcherType
 * @param matches
 */
void matchFramesPairFeatures(
	Mat& firstFrameDescriptor,
	Mat& secondFrame,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
);

#ifdef USE_CUDA
/**
 * For common case.
 *
 * @param firstFrame
 * @param secondFrame
 * @param firstFeatures
 * @param secondFeatures
 * @param matcherType
 * @param matches
 */
void matchFramesPairFeaturesCUDA(
	Mat& firstFrame,
	Mat& secondFrame,
	std::vector<KeyPoint>& firstFeatures,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
);

/**
 * For case when the first descriptor is pre-calculated.
 * 
 * @param firstFrameDescriptor
 * @param secondFrame
 * @param secondFeatures
 * @param matcherType
 * @param matches
 */
void matchFramesPairFeaturesCUDA(
	cuda::GpuMat& firstFrameDescriptor,
	Mat& secondFrame,
	std::vector<KeyPoint>& secondFeatures,
	int matcherType,
	std::vector<DMatch>& matches
);
#endif