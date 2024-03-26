#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../cameraCalibration.h"
#include "../cameraTransition.h"
#include "../fastExtractor.h"
#include "../featureMatching.h"
#include "../IOmisc.h"

#include "../config/config.h"

#include "mainCycle.h"
#include "mainCycleInternals.h"

using namespace cv;


static void defineMediaSources(MediaSources &mediaInputStruct) {
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

static void defineCalibrationMatrix(Mat &calibrationMatrix) {
    // Сreating a new matrix or changing the type and size of an existing one
    calibrationMatrix.create(3, 3, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);
}

static void defineDistortionCoeffs(Mat &distortionCoeffs) {
    // Сreating a new matrix or changing the type and size of an existing one
    distortionCoeffs.create(1, 5, CV_64F);
    loadMatrixFromXML(
		configService.getValue<std::string>(ConfigFieldEnum::CALIBRATION_PATH_).c_str(),
		distortionCoeffs, "DC"
	);
}


void defineProcessingEnvironment(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions)
{
    defineMediaSources(mediaInputStruct);

    defineCalibrationMatrix(dataProcessingConditions.calibrationMatrix);
    defineDistortionCoeffs(dataProcessingConditions.distortionCoeffs);

    dataProcessingConditions.featureExtractingThreshold =
        configService.getValue<int>(ConfigFieldEnum::FEATURE_EXTRACTING_THRESHOLD_);
    dataProcessingConditions.frameBatchSize =
        configService.getValue<int>(ConfigFieldEnum::FRAMES_BATCH_SIZE_);
    dataProcessingConditions.requiredExtractedPointsCount =
        configService.getValue<int>(ConfigFieldEnum::REQUIRED_EXTRACTED_POINTS_COUNT_);
    dataProcessingConditions.requiredMatchedPointsCount =
        configService.getValue<int>(ConfigFieldEnum::REQUIRED_MATCHED_POINTS_COUNT);
    dataProcessingConditions.matcherType = getMatcherTypeIndex();
}


bool getNextFrame(MediaSources &mediaInputStruct, Mat &nextFrame) {
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

///////////////////////////////////////////////////////////////
// Эта функция должна быть либо в main, либо удалить её надо //
///////////////////////////////////////////////////////////////
void defineInitialCameraPosition(TemporalImageData &initialFrame) {
    initialFrame.rotation = Mat::eye(3, 3, CV_64FC1);
    initialFrame.motion = Mat::zeros(3, 1, CV_64FC1);
}


bool findFirstGoodFrame(
    MediaSources &mediaInputStruct, const DataProcessingConditions &dataProcessingConditions,
    Mat &firstGoodFrame, std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (getNextFrame(mediaInputStruct, candidateFrame)) {
        /// TODO: In future we can do UNDISTORTION here
		firstGoodFrame = candidateFrame;
        fastExtractor(firstGoodFrame, goodFrameFeatures, 
            dataProcessingConditions.featureExtractingThreshold);
        
        // Check if enough features are extracted
        if (goodFrameFeatures.size() >= dataProcessingConditions.requiredExtractedPointsCount) {
            return true;
        }
    }

    // No good frame found
    return false;
}


void computeTransformationAndFilterPoints(
    const DataProcessingConditions &dataProcessingConditions, Mat &chiralityMask,
    const TemporalImageData &firstFrameData, TemporalImageData &secondFrameData,
    std::vector<Point2f> &keyPointFrameCoords1, std::vector<Point2f> &keyPointFrameCoords2)
{
	getKeyPointCoordsFromFramePair(firstFrameData.allExtractedFeatures, 
        secondFrameData.allExtractedFeatures, secondFrameData.allMatches, 
        keyPointFrameCoords1, keyPointFrameCoords2);

    estimateTransformation(keyPointFrameCoords1, keyPointFrameCoords2, 
        dataProcessingConditions.calibrationMatrix, secondFrameData.rotation,
        secondFrameData.motion, chiralityMask);

    // Apply the chirality mask to the points from the frames
    filterVectorByMask(keyPointFrameCoords1, chiralityMask);
    filterVectorByMask(keyPointFrameCoords2, chiralityMask);
}


void defineFeaturesCorrespondSpatialIndices(
    const Mat &chiralityMask, TemporalImageData &firstFrameData, TemporalImageData &secondFrameData)
{
    // Resize the correspondence spatial point indices vectors for the previous and new frames
    firstFrameData.correspondSpatialPointIdx.resize(
        firstFrameData.allExtractedFeatures.size(), -1);
    secondFrameData.correspondSpatialPointIdx.resize(
        secondFrameData.allExtractedFeatures.size(), -1);

    int newMatchIdx = 0;
    for (int matchIdx = 0; matchIdx < secondFrameData.allMatches.size(); matchIdx++) {
        // Check if the match is valid based on the chirality mask
        if (chiralityMask.at<uchar>(matchIdx) > 0) {
            // Update correspondence indices for keypoints in the previous and new frames
            firstFrameData.correspondSpatialPointIdx.at(
                secondFrameData.allMatches[matchIdx].queryIdx) = newMatchIdx;
            secondFrameData.correspondSpatialPointIdx.at(
                secondFrameData.allMatches[matchIdx].trainIdx) = newMatchIdx;
            newMatchIdx++;
        }
    }
}


void getOldSpatialPointsAndNewFeatureCoords(
    const std::vector<DMatch> &matches, const std::vector<int> &prevFrameCorrespondIndices,
    const std::vector<Point3f> &allSpatialPoints, const std::vector<KeyPoint> &newFrameKeyPoints, 
    std::vector<Point3f> &oldSpatialPointsForNewFrame, std::vector<Point2f> &newFrameFeatureCoords)
{
    for (auto &match : matches) {
        int structIdx = prevFrameCorrespondIndices[match.queryIdx];
        if (structIdx >= 0) {
            oldSpatialPointsForNewFrame.push_back(allSpatialPoints[structIdx]);
            newFrameFeatureCoords.push_back(newFrameKeyPoints[match.trainIdx].pt);
        }
    }
}


void pushNewSpatialPoints(
	const std::vector<DMatch> &matches, const std::vector<Point3f> &newSpatialPoints,
	std::vector<Point3f> &allSpatialPoints, std::vector<int> &prevFrameCorrespondIndices,
	std::vector<int> &newFrameCorrespondIndices)
{
	for (int i = 0; i < matches.size(); i++) {
		int structIdx = prevFrameCorrespondIndices[matches[i].queryIdx];
        if (structIdx < 0) {
            // If it is a new point in space, add the point to the structure, and the spatial
            // point indices of the pair of matching points are the indexes of the newly added points
            allSpatialPoints.push_back(newSpatialPoints[i]);
            prevFrameCorrespondIndices[matches[i].queryIdx] = ((int)allSpatialPoints.size()) - 1;
            newFrameCorrespondIndices[matches[i].trainIdx] = ((int)allSpatialPoints.size()) - 1;
        } else {
            // If the point already exists in space, the space points 
            // corresponding to the pair of matching points should be the same, with the same index
            newFrameCorrespondIndices[matches[i].trainIdx] = structIdx;
        }
	}
}
