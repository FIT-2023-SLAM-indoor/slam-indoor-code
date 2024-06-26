#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../../calibration/cameraCalibration.h"
#include "../translation/cameraTranslation.h"
#include "../featureExtraction/fastExtractor.h"
#include "../featureMatching/featureMatching.h"
#include "../featureMatching/featureMatchingCommon.h"
#include "../../misc/IOmisc.h"

#include "../../config/config.h"

#include "mainCycleInternals.h"
#include "mainCycleStructures.h"

using namespace cv;


/**
 * Сохраняем цвет трехмерной точки, получая из кадра цвет фичи соответствующего матча.
 *
 * @param [in] frame
 * @param [in] matchIdx идентификатор нужного матча, который прошел chirality mask
 * @param [in, out] frameData
 * @param [out] spatialPointColors
 */
static void saveFrameColorOfKeyPoint(
    const Mat &frame, int matchIdx, TemporalImageData &frameData,
    std::vector<Vec3b> &spatialPointColors)
{
    int frameFeatureIdOfKeyPoint = frameData.allMatches.at(matchIdx).trainIdx;
    Point2f &keyPointFrameCoords = frameData.allExtractedFeatures.at(frameFeatureIdOfKeyPoint).pt;
    spatialPointColors.push_back(frame.at<Vec3b>(keyPointFrameCoords.y, keyPointFrameCoords.x));
}


/**
 * Загружает данные о том, работаем мы с видео или фото, из конфига и сохраняет их в структуру
 * медиа данных.
 *
 * @param [out] mediaInputStruct
 */
static void defineMediaSources(MediaSources &mediaInputStruct) {
    mediaInputStruct.isPhotoProcessing = configService.getValue<bool>(
        ConfigFieldEnum::USE_PHOTOS_CYCLE);

    if (mediaInputStruct.isPhotoProcessing) {
        glob(
            configService.getValue<std::string>(ConfigFieldEnum::PHOTOS_PATH_PATTERN),
            mediaInputStruct.photosPaths, false);
        sortGlobs(mediaInputStruct.photosPaths);
    } else {
        mediaInputStruct.frameSequence.open(
            configService.getValue<std::string>(ConfigFieldEnum::VIDEO_SOURCE_PATH));
        if (!mediaInputStruct.frameSequence.isOpened()) {
            std::cerr << "Video wasn't opened" << std::endl;
            exit(-1);
        }
    }
}

/**
 * Загружает данные о матрице калибровки камеры из конфига.
 *
 * @param [out] distortionCoeffs
 */
static void defineDistortionCoeffs(Mat &distortionCoeffs) {
    // Сreating a new matrix or changing the type and size of an existing one
    distortionCoeffs.create(1, 5, CV_64F);
    loadMatrixFromXML(
        configService.getValue<std::string>(ConfigFieldEnum::CALIBRATION_PATH).c_str(),
        distortionCoeffs, "DC"
    );
}


void defineProcessingEnvironment(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions
) {
    defineMediaSources(mediaInputStruct);
    defineDistortionCoeffs(dataProcessingConditions.distortionCoeffs);

    dataProcessingConditions.featureExtractingThreshold =
        configService.getValue<int>(ConfigFieldEnum::FEATURE_EXTRACTING_THRESHOLD);
	dataProcessingConditions.threadsCount =
		configService.getValue<int>(ConfigFieldEnum::THREADS_COUNT);
    dataProcessingConditions.frameBatchSize =
        configService.getValue<int>(ConfigFieldEnum::FRAMES_BATCH_SIZE);
	dataProcessingConditions.skipFramesFromBatchHead =
		configService.getValue<int>(ConfigFieldEnum::SKIP_FRAMES_FROM_BATCH_HEAD);
	dataProcessingConditions.useFirstFitInBatch =
		configService.getValue<bool>(ConfigFieldEnum::USE_FIRST_FIT_IN_BATCH);
    dataProcessingConditions.requiredExtractedPointsCount =
        configService.getValue<int>(ConfigFieldEnum::REQUIRED_EXTRACTED_POINTS_COUNT);
    dataProcessingConditions.requiredMatchedPointsCount =
        configService.getValue<int>(ConfigFieldEnum::REQUIRED_MATCHED_POINTS_COUNT);
    dataProcessingConditions.matcherType = getMatcherTypeIndex();
	dataProcessingConditions.useBundleAdjustment =
		configService.getValue<bool>(ConfigFieldEnum::USE_BUNDLE_ADJUSTMENT);
	dataProcessingConditions.maxProcessedFramesVectorSz =
		configService.getValue<int>(ConfigFieldEnum::BA_MAX_FRAMES_CNT);
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


void defineCameraPosition(
    const std::deque<TemporalImageData> &oldImageDataDeque, int lastFrameOfLaunchId,
    TemporalImageData &frameData)
{
    if (lastFrameOfLaunchId < 0 || oldImageDataDeque.size() == 0) {
        frameData.rotation = Mat::eye(3, 3, CV_64FC1);
        frameData.motion = Mat::zeros(3, 1, CV_64FC1);
    } else {
        frameData.rotation = oldImageDataDeque.at(lastFrameOfLaunchId).rotation.clone();
        frameData.motion = oldImageDataDeque.at(lastFrameOfLaunchId).motion.clone();
    }
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
    const DataProcessingConditions &dataProcessingConditions, Mat &calibrationMatrix,
    const TemporalImageData &firstFrameData, TemporalImageData &secondFrameData,
    std::vector<Point2f> &keyPointFrameCoords1, std::vector<Point2f> &keyPointFrameCoords2,
	Mat &chiralityMask
) {
    getKeyPointCoordsFromFramePair(firstFrameData.allExtractedFeatures,
        secondFrameData.allExtractedFeatures, secondFrameData.allMatches,
        keyPointFrameCoords1, keyPointFrameCoords2);

    estimateTransformation(keyPointFrameCoords1, keyPointFrameCoords2,
        calibrationMatrix, secondFrameData.rotation,
        secondFrameData.motion, chiralityMask);

    // Apply the chirality mask to the points from the frames
    filterVectorByMask(keyPointFrameCoords1, chiralityMask);
    filterVectorByMask(keyPointFrameCoords2, chiralityMask);
}


void defineFeaturesCorrespondSpatialIndices(
    const Mat &chiralityMask, const Mat &secondFrame, TemporalImageData &firstFrameData,
    TemporalImageData &secondFrameData, std::vector<Vec3b> &firstPairSpatialPointColors)
{
    // Resize the correspondence spatial point indices vectors for the first and second frames
    firstFrameData.correspondSpatialPointIdx.resize(
        firstFrameData.allExtractedFeatures.size(), -1);
    secondFrameData.correspondSpatialPointIdx.resize(
        secondFrameData.allExtractedFeatures.size(), -1);

    int newMatchIdx = 0;
    for (int matchIdx = 0; matchIdx < secondFrameData.allMatches.size(); matchIdx++) {
        // Check if the match is valid based on the chirality mask
        if (chiralityMask.at<uchar>(matchIdx) > 0) {
            // Update correspondence indices for keypoints in the first and second frames
            firstFrameData.correspondSpatialPointIdx.at(
                secondFrameData.allMatches[matchIdx].queryIdx) = newMatchIdx;
            secondFrameData.correspondSpatialPointIdx.at(
                secondFrameData.allMatches[matchIdx].trainIdx) = newMatchIdx;
            // Save color of current good key point (with index = newMatchIdx)
            saveFrameColorOfKeyPoint(secondFrame, matchIdx, secondFrameData,
                firstPairSpatialPointColors);
            // Update index for next good matches
            newMatchIdx++;
        }
    }
}


void getOldSpatialPointsAndNewFrameFeatureCoords(
    const std::vector<DMatch> &matches, const std::vector<int> &prevFrameCorrespondIndices,
    const SpatialPointsVector &allSpatialPoints, const std::vector<KeyPoint> &newFrameKeyPoints,
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
    const Mat &newFrame, const SpatialPointsVector &newSpatialPoints,
    GlobalData &globalDataStruct, std::vector<int> &prevFrameCorrespondIndices,
    TemporalImageData &newFrameData)
{
    newFrameData.correspondSpatialPointIdx.resize(newFrameData.allExtractedFeatures.size(), -1);
    for (int i = 0; i < newFrameData.allMatches.size(); i++) {
        int structId = prevFrameCorrespondIndices[newFrameData.allMatches[i].queryIdx];
        if (structId < 0) {
            // If it is a new point in space, add the point to the structure, and the spatial
            // point indices of the pair of matching points are the indexes of the newly added points
            globalDataStruct.spatialPoints.push_back(newSpatialPoints[i]);
            saveFrameColorOfKeyPoint(newFrame, i, newFrameData,
                globalDataStruct.spatialPointsColors);
            prevFrameCorrespondIndices[newFrameData.allMatches[i].queryIdx] =
                globalDataStruct.spatialPoints.size() - 1;
            newFrameData.correspondSpatialPointIdx[newFrameData.allMatches[i].trainIdx] =
                globalDataStruct.spatialPoints.size() - 1;
        } else {
            // If the point already exists in space, the space points
            // corresponding to the pair of matching points should be the same, with the same index
            newFrameData.correspondSpatialPointIdx[newFrameData.allMatches[i].trainIdx] = structId;
        }
    }
}


void insertNewGlobalData(GlobalData &mainGlobalData, GlobalData &newGlobalData) {
    for (int pointId = 0; pointId < newGlobalData.spatialPoints.size(); pointId++) {
        mainGlobalData.spatialPoints.push_back(newGlobalData.spatialPoints.at(pointId));
        mainGlobalData.spatialPointsColors.push_back(
            newGlobalData.spatialPointsColors.at(pointId));
    }
    for (int cameraId = 0; cameraId < newGlobalData.cameraRotations.size(); cameraId++) {
        mainGlobalData.cameraRotations.push_back(
            newGlobalData.cameraRotations.at(cameraId).clone());
        mainGlobalData.spatialCameraPositions.push_back(
            newGlobalData.spatialCameraPositions.at(cameraId).clone());
    }
}


void checkGlobalDataStruct(GlobalData &globalDataStruct) {
    if (globalDataStruct.spatialPoints.empty()) {
        logStreams.mainReportStream << "Couldn't process image sequence. Too little data.\n";
        logStreams.mainReportStream.flush();
        exit(-1);
    }
}