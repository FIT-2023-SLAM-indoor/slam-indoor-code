#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "thread"
#include "chrono"
#include "atomic"

#include "../cameraTransition.h"
#include "../fastExtractor.h"
#include "../featureMatching.h"
#include "../triangulate.h"
#include "../IOmisc.h"
#include "../bundleAdjustment/bundleAdjustment.h"

#include "../config/config.h"

#include "mainCycle.h"
#include "mainCycleInternals.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;

/**
 * Fills batch up to dataProcessingConditions.batchSize or while new frames can be obtained.
 * <br>
 * Only frames with extracted points count greater or equal to
 * dataProcessingConditions.requiredExtractedPointsCount will be added to batch
 *
 * @param [in] mediaInputStruct
 * @param [in] dataProcessingConditions
 * @param [in,out] currentBatch batch can contains some frames which was considered as bad for previous frame
 *     but can be good for new one
 *
 * @return skipped frames count
 */
static int fillVideoFrameBatch(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch
);

#define EMPTY_BATCH (-2)
#define FRAME_NOT_FOUND (-1)
/**
 * Find new frame from batch which can be up to dataProcessingConditions.batchSize
 *
 * @param [in] mediaInputStruct
 * @param [in] dataProcessingConditions
 * @param [in,out] currentBatch <ul>
 * 		<li>Input is a batch tail from previous iteration</li>
 * 		<li>
 * 			Output is a batch tail after new good frame
 * 			(consequently there will be full batch in case we didn't find new good one)
* 		</li>
 * </ul>
 * @param [in] previousFrame
 * @param [out] newGoodFrame
 * @param [in] previousFeatures
 * @param [out] newFeatures
 * @param [out] matches
 * @return found frame batch index or some of error codes described above
 */
static int findGoodFrameFromBatch(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame,
	Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);


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
		batchFrameIndex = findGoodFrameFromBatch(
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
		frameIndex = findGoodFrameFromBatch(
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


static int fillVideoFrameBatch(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch
) {
	std::clock_t start = clock();
	int currentFrameBatchSize = 0;
	Mat nextFrame;

	std::vector<KeyPoint> nextFeatures;
	int skippedFrames = 0, skippedFramesForFirstFound = 0;
	logStreams.mainReportStream << "Features count in frames added to batch: ";
	while (
		currentBatch.size() < dataProcessingConditions.frameBatchSize
		&& getNextFrame(mediaInputStruct, nextFrame)
	) {
		// TODO: here can be undistortion
		fastExtractor(nextFrame, nextFeatures,
					  dataProcessingConditions.featureExtractingThreshold);
		if (nextFeatures.size() < dataProcessingConditions.requiredExtractedPointsCount) {
			if (currentFrameBatchSize == 0)
				skippedFramesForFirstFound++;
			skippedFrames++;
			continue;
		}
		logStreams.mainReportStream << nextFeatures.size() << " ";
		currentBatch.push_back({
			nextFrame.clone(),
			nextFeatures
		});
		currentFrameBatchSize++;
	}
	logStreams.mainReportStream << std::endl << "Skipped for first: " << skippedFramesForFirstFound << std::endl;
	logStreams.mainReportStream << "Skipped frames while constructing batch: " << skippedFrames << std::endl;
	logStreams.mainReportStream << "Batch size: " << currentBatch.size() << std::endl;
	std::cout << "Extracting time: " << (clock() - start) / CLOCKS_PER_SEC << std::endl;
	return skippedFrames;
}


static int findGoodFrameFromBatch(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
) {
	int skippedFramesCount =
		fillVideoFrameBatch(mediaInputStruct, dataProcessingConditions, currentBatch);
	int currentBatchSz = currentBatch.size();
	if (currentBatchSz == 0)
		return EMPTY_BATCH;

	logStreams.mainReportStream << "Prev. extracted: " << previousFeatures.size() << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	// In the future this must
//	std::vector<std::atomic_flag> isMatchesEstimated(currentBatchSz);
	std::vector<bool> isMatchesEstimated(currentBatchSz, false);
	int threadsCnt = dataProcessingConditions.threadsCount;
	std::vector<std::thread> threads;
	std::vector<bool> threadShouldDieFlags(threadsCnt, false);
	std::vector<std::vector<DMatch>> estimatedMatches(currentBatchSz, std::vector<DMatch>());
	for (int i = 0; i < threadsCnt; ++i) {
		threads.emplace_back(std::thread([&](int threadIndex) {
			for (
				int batchIndex = currentBatchSz - 1 - threadIndex;
				batchIndex >= 0;
				batchIndex -= threadsCnt
			) {
				matchFramesPairFeatures(previousFrame, currentBatch.at(batchIndex).frame,
					previousFeatures, currentBatch.at(batchIndex).features,
					dataProcessingConditions.matcherType, estimatedMatches.at(batchIndex));
				isMatchesEstimated.at(batchIndex) = true;
				if (threadShouldDieFlags.at(threadIndex))
					break;
			}
		}, i));
	}

	Mat goodFrame;
	std::vector<KeyPoint> goodFeatures;
	std::vector<DMatch> goodMatches;
	int goodIndex = FRAME_NOT_FOUND;
	Mat candidateFrame;
	std::vector<KeyPoint> candidateFrameFeatures;
	std::vector<DMatch> candidateMatches;
	for (
		int batchIndex = currentBatchSz - 1;
		batchIndex >= dataProcessingConditions.skipFramesFromBatchHead;
		batchIndex--
	) {
		while (!isMatchesEstimated.at(batchIndex)); // Dmitry Valentinovich, I'm sorry...
		candidateFrame = currentBatch.at(batchIndex).frame.clone();
		candidateFrameFeatures = currentBatch.at(batchIndex).features;
		candidateMatches = estimatedMatches.at(batchIndex);

		logStreams.mainReportStream << "Batch index: " << batchIndex
									<< "; curr. extracted: " << candidateFrameFeatures.size()
									<< "; matched " << candidateMatches.size() << std::endl;

		if (
			candidateMatches.size() >= dataProcessingConditions.requiredMatchedPointsCount
			&& candidateMatches.size() >= goodMatches.size()
		) {
			logStreams.mainReportStream << "Frame " << batchIndex << " is a good" << std::endl;
			goodIndex = batchIndex;
			goodFrame = candidateFrame.clone();
			goodFeatures = candidateFrameFeatures;
			goodMatches = candidateMatches;
			if (dataProcessingConditions.useFirstFitInBath)
				break;
		}
	}
	std::cout << "Matching time for index " << goodIndex << ": "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start
		).count() << std::endl;
	for (int i = threadShouldDieFlags.size() - 1; i >= 0; --i)
		threadShouldDieFlags.at(i) = true;
	for (auto& thread : threads)
		thread.join();
	std::cout << "Threads terminated: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(
				  std::chrono::high_resolution_clock::now() - start
			  ).count() << std::endl;
	if (goodIndex != FRAME_NOT_FOUND) {
		newGoodFrame = goodFrame.clone();
		newFeatures = goodFeatures;
		matches = goodMatches;

		std::vector<BatchElement> batchTail;
		for (int i = goodIndex + 1; i < currentBatchSz; ++i)
			batchTail.push_back(currentBatch.at(i));
		currentBatch = batchTail;

		logStreams.extractedMatchedTable << skippedFramesCount << "; "
										 << goodIndex << "; "
										 << previousFeatures.size() << "; "
										 << goodFeatures.size() << "; "
										 << matches.size() << ";\n";
		return goodIndex;
	}

	return goodIndex;
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
