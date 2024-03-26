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
    Mat &goodFrame, std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (getNextFrame(mediaInputStruct, candidateFrame)) {
        /// TODO: In future we can do UNDISTORTION here
		goodFrame = candidateFrame;
        fastExtractor(goodFrame, goodFrameFeatures, 
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
    const TemporalImageData &prevFrameData, TemporalImageData &newFrameData,
    std::vector<Point2f> &keyPointFrameCoords1, std::vector<Point2f> &keyPointFrameCoords2)
{
	getKeyPointCoordsFromFramePair(prevFrameData.allExtractedFeatures, 
        newFrameData.allExtractedFeatures, newFrameData.allMatches, 
        keyPointFrameCoords1, keyPointFrameCoords2);

    estimateTransformation(keyPointFrameCoords1, keyPointFrameCoords2, 
        dataProcessingConditions.calibrationMatrix,newFrameData.rotation,
        newFrameData.motion, chiralityMask);

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


void getObjAndImgPoints(
    const std::vector<DMatch> &matches, const std::vector<int> &correspondSpatialPointIdx,
    const std::vector<Point3f> &spatialPoints, const std::vector<KeyPoint> &extractedFeatures,
    std::vector<Point3f> &objPoints, std::vector<Point2f> &imgPoints)
{
    for (auto &match : matches) {
        int structIdx = correspondSpatialPointIdx[match.queryIdx];
        if (structIdx >= 0) {
            objPoints.push_back(spatialPoints[structIdx]);
            imgPoints.push_back(extractedFeatures[match.trainIdx].pt);
        }
    }
}


/**
 * Написать документацию!!!
*/
void pushNewSpatialPoints(
	const std::vector<DMatch> &matches,
	std::vector<int> &prevFrameCorrespondingIndices,
	std::vector<int> &currFrameCorrespondingIndices,
	const std::vector<Point3f> &newSpatialPoints,
	std::vector<Point3f> &allSpatialPoints)
{
	for (int i = 0; i < matches.size(); i++) {
		int structIdx = prevFrameCorrespondingIndices[matches[i].queryIdx];
        // If the point already exists in space, the space points 
        // corresponding to the pair of matching points should be the same, with the same index
		if (structIdx >= 0) {
			currFrameCorrespondingIndices[matches[i].trainIdx] = structIdx;
			continue;
		}

		// If the point already exists in space, add the point to the structure, and the spatial
        // point indexes of the pair of matching points are the indexes of the newly added points
		allSpatialPoints.push_back(newSpatialPoints[i]);
		prevFrameCorrespondingIndices[matches[i].queryIdx] = ((int)allSpatialPoints.size()) - 1;
		currFrameCorrespondingIndices[matches[i].trainIdx] = ((int)allSpatialPoints.size()) - 1;
	}
}
