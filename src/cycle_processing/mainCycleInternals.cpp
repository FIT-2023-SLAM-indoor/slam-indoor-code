#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "fastExtractor.h"
#include "featureMatching.h"
#include "triangulate.h"
#include "IOmisc.h"

#include "config/config.h"

#include "mainCycle.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;


void defineCalibrationMatrix(Mat &calibrationMatrix) {
    // Сreating a new matrix or changing the type and size of an existing one
    calibrationMatrix.create(3, 3, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);
}


void defineDistortionCoeffs(Mat &distortionCoeffs) {
    // Сreating a new matrix or changing the type and size of an existing one
    distortionCoeffs.create(1, 5, CV_64F);
    loadMatrixFromXML(
		configService.getValue<std::string>(ConfigFieldEnum::CALIBRATION_PATH_).c_str(),
		distortionCoeffs, "DC"
	);
}


void defineMediaSources(MediaSources &mediaInputStruct) {
    mediaInputStruct.isPhotoProcessing = configService.getValue<bool>(
        ConfigFieldEnum::USE_PHOTOS_CYCLE);

    if (mediaInputStruct.isPhotoProcessing) {
		glob(
            configService.getValue<std::string>(ConfigFieldEnum::PHOTOS_PATH_PATTERN_), 
            mediaInputStruct.photosPaths, false);
		sortGlobs(mediaInputStruct.photosPaths);
    } else {
        mediaInputStruct.frameSequence.open(
            configService.getValue<std::string>(ConfigFieldEnum::VIDEO_SOURCE_PATH_));
        if (!mediaInputStruct.frameSequence.isOpened()) {
			std::cerr << "Video wasn't opened" << std::endl;
			exit(-1);
		}
    }
}


void defineProcessingEnvironment(DataProcessingConditions &dataProcessingConditions) {
    defineCalibrationMatrix(dataProcessingConditions.calibrationMatrix);
    defineDistortionCoeffs(dataProcessingConditions.distortionCoeffs);

    configService.getValue<int>(ConfigFieldEnum::FRAMES_BATCH_SIZE_),

    dataProcessingConditions.featureExtractingThreshold =
        configService.getValue<int>(ConfigFieldEnum::FEATURE_EXTRACTING_THRESHOLD_);
    dataProcessingConditions.requiredExtractedPointsCount =
        configService.getValue<int>(ConfigFieldEnum::REQUIRED_EXTRACTED_POINTS_COUNT_);
    dataProcessingConditions.requiredMatchedPointsCount =
        configService.getValue<int>(ConfigFieldEnum::REQUIRED_MATCHED_POINTS_COUNT);
    dataProcessingConditions.matcherType = getMatcherTypeIndex();
}


bool getNextFrame(MediaSources &mediaInputStruct, Mat& nextFrame) {
    if (mediaInputStruct.isPhotoProcessing) {
        if (mediaInputStruct.photosPaths.empty()) {
            return false;
        }
        nextFrame = imread(mediaInputStruct.photosPaths.front());
        auto iter = mediaInputStruct.photosPaths.cbegin();
        mediaInputStruct.photosPaths.erase(iter);
        return !nextFrame.empty();
    } else {
        return mediaInputStruct.frameSequence.read(nextFrame);
    }
}


void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    DataProcessingConditions &dataProcessingConditions,
    std::vector<DMatch>& matches)
{
    // Extract descriptors from the key points of the input frames
    Mat firstDescriptor;
    extractDescriptor(firstFrame, firstFeatures, 
        dataProcessingConditions.matcherType, firstDescriptor);
    Mat secondDescriptor;
    extractDescriptor(secondFrame, secondFeatures, 
        dataProcessingConditions.matcherType, secondDescriptor);

    // Match the descriptors using the specified matcher type and radius
    matchFeatures(firstDescriptor, secondDescriptor, matches, 
        dataProcessingConditions.matcherType, dataProcessingConditions.radius);
}


void getObjAndImgPoints(
    std::vector<DMatch> &matches,
    std::vector<int> &correspondSpatialPointIdx,
    std::vector<Point3f> &spatialPoints,
    std::vector<KeyPoint> &extractedFeatures,
    std::vector<Point3f> &objPoints,
    std::vector<Point2f> &imgPoints)
{
    for (auto &match : matches) {
        int query_idx = match.queryIdx;
        int train_idx = match.trainIdx;

        int struct_idx = correspondSpatialPointIdx[query_idx];
        if (struct_idx >= 0) {
            objPoints.push_back(spatialPoints[struct_idx]);
            imgPoints.push_back(extractedFeatures[train_idx].pt);
        }
    }
}


void getMatchedPointCoords(
	std::vector<KeyPoint> &firstExtractedFeatures,
	std::vector<KeyPoint> &secondExtractedFeatures,
	std::vector<DMatch> &matches,
	std::vector<Point2f> &firstMatchedPoints,
	std::vector<Point2f> &secondMatchedPoints)
{
	firstMatchedPoints.clear();
	secondMatchedPoints.clear();
	for (int i = 0; i < matches.size(); i++) {
		firstMatchedPoints.push_back(firstExtractedFeatures[matches[i].queryIdx].pt);
		secondMatchedPoints.push_back(secondExtractedFeatures[matches[i].trainIdx].pt);
	}
}


void initTemporalImageDataDeque(std::deque<TemporalImageData> &temporalImageDataDeque) {
    for (int imageDataIdx = 0; imageDataIdx < temporalImageDataDeque.size(); imageDataIdx++) {
        temporalImageDataDeque.at(imageDataIdx).allExtractedFeatures.clear();
        temporalImageDataDeque.at(imageDataIdx).allMatches.clear();
        temporalImageDataDeque.at(imageDataIdx).correspondSpatialPointIdx.clear();
    }
}