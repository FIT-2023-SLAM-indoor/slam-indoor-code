#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "../cameraTransition.h"
#include "../featureMatching.h"
#include "../triangulate.h"
#include "../IOmisc.h"
#include "../bundleAdjustment/bundleAdjustment.h"

#include "../config/config.h"

#include "mainCycle.h"
#include "mainCycleInternals.h"
#include "batch.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;

/**
 * Функция по умному ищет подходящие для реконструкции первые два кадра. Если второй хороший кадр
 * не удалось найти сразу, то мы вынимаем первый кадр из батча, но при этом добавляем в батч
 * следующее изображение из последовательности с достаточным количеством фич.
 * Функция в качестве числа вернет либо индекс второго хорошего кадра из батча, либо константу о
 * том, что видео кончилось, других варианов нет. (*Эта функция нужна как раз для исключения
 * варианта не нахождения в батче второго кадра при поиске первой пары).
 *
 * @param [in] dataProcessingConditions
 * @param [in] mediaInputStruct
 * @param [in, out] currentBatch
 * @param [out] temporalImageDataDeque
 * @param [out] secondFrame
 * @return EMPTY_BATCH or frameIndex.
 */
static int defineFirstPairFrames( 
	const DataProcessingConditions &dataProcessingConditions,
	MediaSources &mediaInputStruct,
	std::vector<BatchElement> &currentBatch, 
	std::deque<TemporalImageData> &temporalImageDataDeque,
	Mat &secondFrame
);


/**
 *
 */
static int processingFirstPairFrames(
	MediaSources &mediaInputStruct,
	Mat &calibrationMatrix,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	std::deque<TemporalImageData> &temporalImageDataDeque,
	Mat &secondFrame, SpatialPointsVector &spatialPoints,
	std::vector<cv::Vec3b> &firstPairSpatialPointColors
);

/**
 * Adds rotations and motions to global struct and clears processed frames datas' vector.
 *
 * @param [in,out] processedFramesData
 * @param [in,out] globalDataStruct
 * @param [in] bundleAdjustmentWasUsed if true, print saved rotations and motions
 */
static void moveProcessedDataToGlobalStruct(
	std::vector<TemporalImageData> &processedFramesData,
	GlobalData &globalDataStruct,
	bool bundleAdjustmentWasUsed = false
);


int mainCycle(
	MediaSources &mediaInputStruct, Mat &calibrationMatrix,
	const DataProcessingConditions &dataProcessingConditions,
	std::deque<TemporalImageData> &temporalImageDataDeque, GlobalData &globalDataStruct
) {
	logStreams.mainReportStream << "Launching main cycle..." << std::endl;

	std::vector<BatchElement> batch; // Необходимо создавать батч тут, чтобы его не использованный хвост переносился на следующую итерацию
	std::vector<TemporalImageData> processedFramesData; // Used for BA or just for saving data to global struct

    Mat lastGoodFrame;
    if (processingFirstPairFrames(
		mediaInputStruct, calibrationMatrix, dataProcessingConditions, batch,
		temporalImageDataDeque, lastGoodFrame, globalDataStruct.spatialPoints,
		globalDataStruct.spatialPointsColors) == EMPTY_BATCH
	) {
        // Error message if at least two good frames are not found in the video
        return EMPTY_BATCH;
    }
	processedFramesData.push_back(temporalImageDataDeque.at(0));
	processedFramesData.push_back(temporalImageDataDeque.at(1));

	logStreams.mainReportStream << "The first frame transformation:" << std::endl;
	logStreams.mainReportStream << temporalImageDataDeque.at(0).rotation << std::endl;
	logStreams.mainReportStream << temporalImageDataDeque.at(0).motion.t() << std::endl;
	logStreams.mainReportStream << "The second frame transformation:" << std::endl;
	logStreams.mainReportStream << temporalImageDataDeque.at(1).rotation << std::endl;
	logStreams.mainReportStream << temporalImageDataDeque.at(1).motion.t() << std::endl;
	rawOutput(temporalImageDataDeque.at(0).motion.t(), logStreams.poseStream);
	rawOutput(temporalImageDataDeque.at(1).motion.t(), logStreams.poseStream);
	logStreams.mainReportStream.flush();
	logStreams.poseStream.flush();

	int batchFrameIndex;
    int lastFrameIdx = 1;
    while (true) {
        logStreams.mainReportStream << std::endl << "================================================================\n" << std::endl;

        // Find the next good frame batch
        Mat nextGoodFrame;
		batchFrameIndex = findGoodFrameFromBatchMultithreadingWrapper(
				mediaInputStruct, dataProcessingConditions, batch,
				lastGoodFrame, nextGoodFrame,
				temporalImageDataDeque.at(lastFrameIdx).allExtractedFeatures,
				temporalImageDataDeque.at(lastFrameIdx+1).allExtractedFeatures,
				temporalImageDataDeque.at(lastFrameIdx+1).allMatches
		);
        if (batchFrameIndex == EMPTY_BATCH) {
			break;
        } else if (batchFrameIndex == FRAME_NOT_FOUND) {
			std::string msg = "No good frames in batch. Interrupt video processing";
			logStreams.mainReportStream << msg << std::endl;
			std::cerr << msg << std::endl;
            break;
        }

        // Get object and image points for reconstruction
        std::vector<Point3f> oldSpatialPointsForNewFrame;
        std::vector<Point2f> newFrameFeatureCoords;
        getOldSpatialPointsAndNewFrameFeatureCoords(
            temporalImageDataDeque.at(lastFrameIdx+1).allMatches,
            temporalImageDataDeque.at(lastFrameIdx).correspondSpatialPointIdx,
            globalDataStruct.spatialPoints,
            temporalImageDataDeque.at(lastFrameIdx+1).allExtractedFeatures,
            oldSpatialPointsForNewFrame, newFrameFeatureCoords
		);

        // Find transformation matrix
        if (oldSpatialPointsForNewFrame.size() < 4) {
            logStreams.mainReportStream << "Not enough corresponding points for solvePnP RANSAC" << std::endl;
            break;
        }
        Mat rotationVector;
        solvePnPRansac(
			oldSpatialPointsForNewFrame, newFrameFeatureCoords, calibrationMatrix,
			dataProcessingConditions.distortionCoeffs, rotationVector,
			temporalImageDataDeque.at(lastFrameIdx+1).motion
		);
        // Convert rotation vector to rotation matrix
        Rodrigues(rotationVector, temporalImageDataDeque.at(lastFrameIdx+1).rotation);

        logStreams.mainReportStream << "Used in solvePnP: " << oldSpatialPointsForNewFrame.size() << std::endl;
        logStreams.mainReportStream << temporalImageDataDeque.at(lastFrameIdx+1).rotation << std::endl;
        logStreams.mainReportStream << temporalImageDataDeque.at(lastFrameIdx+1).motion.t() << std::endl;
        logStreams.mainReportStream.flush();
        rawOutput(temporalImageDataDeque.at(lastFrameIdx+1).motion.t(), logStreams.poseStream);
        logStreams.poseStream.flush();

        // Get matched point coordinates for reconstruction
        std::vector<Point2f> matchedPointCoords1;
        std::vector<Point2f> matchedPointCoords2;
		SpatialPointsVector newSpatialPoints;
        getKeyPointCoordsFromFramePair(
            temporalImageDataDeque.at(lastFrameIdx).allExtractedFeatures,
            temporalImageDataDeque.at(lastFrameIdx+1).allExtractedFeatures,
            temporalImageDataDeque.at(lastFrameIdx+1).allMatches,
            matchedPointCoords1, matchedPointCoords2);
        reconstruct(calibrationMatrix,
            temporalImageDataDeque.at(lastFrameIdx).rotation,
            temporalImageDataDeque.at(lastFrameIdx).motion,
            temporalImageDataDeque.at(lastFrameIdx+1).rotation,
            temporalImageDataDeque.at(lastFrameIdx+1).motion,
            matchedPointCoords1, matchedPointCoords2, newSpatialPoints);
        pushNewSpatialPoints(nextGoodFrame, newSpatialPoints, globalDataStruct,
            temporalImageDataDeque.at(lastFrameIdx).correspondSpatialPointIdx,
            temporalImageDataDeque.at(lastFrameIdx+1));

		processedFramesData.push_back(temporalImageDataDeque.at(lastFrameIdx+1));
		if (processedFramesData.size() >= dataProcessingConditions.maxProcessedFramesVectorSz) {
			if (dataProcessingConditions.useBundleAdjustment)
				bundleAdjustment(calibrationMatrix, processedFramesData, globalDataStruct);
			moveProcessedDataToGlobalStruct(
				processedFramesData, globalDataStruct, dataProcessingConditions.useBundleAdjustment
			);
		}

		// Update last good frame
		lastGoodFrame = nextGoodFrame.clone();
		if (lastFrameIdx == OPTIMAL_DEQUE_SIZE - 2) {
			temporalImageDataDeque.pop_front();
			temporalImageDataDeque.push_back({});
		} else {
			lastFrameIdx++;
		}
    }

	if (!processedFramesData.empty()) {
		if (dataProcessingConditions.useBundleAdjustment) {
			bundleAdjustment(calibrationMatrix, processedFramesData, globalDataStruct);
		}
		moveProcessedDataToGlobalStruct(
			processedFramesData, globalDataStruct, dataProcessingConditions.useBundleAdjustment
		);
	}

	if (batchFrameIndex == EMPTY_BATCH) {
		std::string msg = "Video is over. No more frames...";
		logStreams.mainReportStream << msg << std::endl;
		std::cerr << msg << std::endl;
		return EMPTY_BATCH;
	}
	return lastFrameIdx;
}


static int processingFirstPairFrames(
	MediaSources &mediaInputStruct,
	Mat &calibrationMatrix,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	std::deque<TemporalImageData> &temporalImageDataDeque,
	Mat &secondFrame, SpatialPointsVector &spatialPoints,
	std::vector<cv::Vec3b> &firstPairSpatialPointColors
) {
	int frameIndex = defineFirstPairFrames(dataProcessingConditions, mediaInputStruct, currentBatch,
										   temporalImageDataDeque, secondFrame);
	if (frameIndex == EMPTY_BATCH) {
		return EMPTY_BATCH;
	}

	Mat chiralityMask;
	std::vector<Point2f> extractedPointCoords1, extractedPointCoords2;
	computeTransformationAndFilterPoints(dataProcessingConditions, calibrationMatrix,
										 temporalImageDataDeque.at(0),temporalImageDataDeque.at(1),
										 extractedPointCoords1, extractedPointCoords2, chiralityMask);
	logStreams.mainReportStream << "Second transformation before refinement:" << std::endl;
	logStreams.mainReportStream << temporalImageDataDeque.at(1).rotation << std::endl;
	logStreams.mainReportStream << temporalImageDataDeque.at(1).motion.t() << std::endl;
	refineTransformationForGlobalCoords(
		temporalImageDataDeque.at(0).rotation, temporalImageDataDeque.at(0).motion,
		temporalImageDataDeque.at(1).rotation, temporalImageDataDeque.at(1).motion
	); // Places relative second matrix to global coords. Essential for restarted cycle
	reconstruct(calibrationMatrix,
				temporalImageDataDeque.at(0).rotation, temporalImageDataDeque.at(0).motion,
				temporalImageDataDeque.at(1).rotation, temporalImageDataDeque.at(1).motion,
				extractedPointCoords1, extractedPointCoords2, spatialPoints);
	defineFeaturesCorrespondSpatialIndices(chiralityMask, secondFrame, temporalImageDataDeque.at(0),
										   temporalImageDataDeque.at(1), firstPairSpatialPointColors);

	return frameIndex;
}


static int defineFirstPairFrames(
	const DataProcessingConditions &dataProcessingConditions, MediaSources &mediaInputStruct,
	std::vector<BatchElement> &currentBatch, std::deque<TemporalImageData> &temporalImageDataDeque,
	Mat &secondFrame)
{
	Mat firstFrame;
	if (!findFirstGoodFrame(
		mediaInputStruct, dataProcessingConditions,firstFrame,
		temporalImageDataDeque.at(0).allExtractedFeatures
	)) {
		return EMPTY_BATCH;  // Video is over...
	}

	int frameIndex = FRAME_NOT_FOUND;
	while (frameIndex == FRAME_NOT_FOUND) {
		frameIndex = findGoodFrameFromBatchMultithreadingWrapper(
			mediaInputStruct, dataProcessingConditions, currentBatch,
			firstFrame, secondFrame,
			temporalImageDataDeque.at(0).allExtractedFeatures,
			temporalImageDataDeque.at(1).allExtractedFeatures,
			temporalImageDataDeque.at(1).allMatches
		);
		if (frameIndex == EMPTY_BATCH || frameIndex >= 0) {
			return frameIndex;
		}

		firstFrame = currentBatch.at(0).frame.clone();
		temporalImageDataDeque.at(0).allExtractedFeatures = currentBatch.at(0).features;
		auto iter = currentBatch.cbegin();
        currentBatch.erase(iter);
	}
}

static void moveProcessedDataToGlobalStruct(
	std::vector<TemporalImageData> &processedFramesData,
	GlobalData &globalDataStruct,
	bool bundleAdjustmentWasUsed
) {
	logStreams.mainReportStream << (bundleAdjustmentWasUsed ? "Projections after BA:\n" : "");
	for (int i = 0; i < processedFramesData.size(); ++i) {
		auto &frameData = processedFramesData.at(i);
		globalDataStruct.cameraRotations.push_back(frameData.rotation.clone());
		globalDataStruct.spatialCameraPositions.push_back(frameData.motion.clone());
		if (bundleAdjustmentWasUsed) {
			logStreams.mainReportStream
				<< "Processed frame " << i << ": "
				<< "\nRotation:\n" << frameData.rotation
				<< "\nMotion:\n" << frameData.motion
				<< "\n\n";
		}
	}

	processedFramesData.clear();
}
