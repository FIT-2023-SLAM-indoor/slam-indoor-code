#ifdef USE_CUDA
	#include "opencv2/core/cuda.hpp"
#endif
#include "thread"

#include "../misc/ChronoTimer.h"
#include "../IOmisc.h"
#include "../fastExtractor.h"
#include "../featureMatching.h"

#include "mainCycleInternals.h"

#include "batch.h"

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
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	std::vector<bool> &isMatchesEstimated,
	std::vector<std::vector<DMatch>> &estimatedMatches,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);

int findGoodFrameFromBatchMultithreadingWrapper(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
) {
	fillVideoFrameBatch(mediaInputStruct, dataProcessingConditions, currentBatch);
	int currentBatchSz = currentBatch.size();
	if (currentBatchSz == 0)
		return EMPTY_BATCH;

	logStreams.mainReportStream << "Prev. extracted: " << previousFeatures.size() << std::endl;
	ChronoTimer timer;

	int threadsCnt = dataProcessingConditions.threadsCount;
	std::vector<std::thread> threads;
	std::vector<bool> isMatchesEstimated(currentBatchSz, false);
	volatile bool threadsShouldDie = false;

	std::vector<std::vector<DMatch>> estimatedMatches(currentBatchSz, std::vector<DMatch>());
	Mat previousDescriptor;
	extractDescriptor(previousFrame, previousFeatures,
		dataProcessingConditions.matcherType, previousDescriptor);

	for (int i = threadsCnt - 1; i >= 0; --i) {
		threads.emplace_back(std::thread([&](int threadIndex) {
			for (
				int batchIndex = currentBatchSz - 1 - threadIndex;
				batchIndex >= 0;
				batchIndex -= threadsCnt
			) {
				std::cout << "Start: " << batchIndex << std::endl;
				BatchElement &element = currentBatch.at(batchIndex);
#ifdef USE_CUDA
				cuda::GpuMat previousDescriptorGpu(previousDescriptor);
				matchFramesPairFeaturesCUDA(
					previousDescriptorGpu, element.frame, element.features,
					dataProcessingConditions.matcherType, estimatedMatches.at(batchIndex)
				);
#else
				matchFramesPairFeatures(
					previousDescriptor, element.frame, element.features,
					dataProcessingConditions.matcherType, estimatedMatches.at(batchIndex)
				);
#endif
				isMatchesEstimated.at(batchIndex) = true;
				std::cout << "Finish: " << batchIndex << std::endl;
				this_thread::yield();
				if (threadsShouldDie)
					break;
			}
		}, i));
	}

	timer.printLastPointDelta("Matching threads started at: ", logStreams.timeStream);
	timer.updateLastPoint();

	int returnCode = findGoodFrameFromBatch(
		dataProcessingConditions,
		currentBatch, isMatchesEstimated, estimatedMatches,
		previousFrame, newGoodFrame,
		previousFeatures, newFeatures,
		matches
	);

	logStreams.timeStream << "Matching time for index " << returnCode;
	timer.printLastPointDelta(" : ", logStreams.timeStream);
	timer.updateLastPoint();

	threadsShouldDie = true;
	for (auto& thread : threads)
		thread.join();

	timer.printLastPointDelta("Threads terminated: ", logStreams.timeStream);
	timer.updateLastPoint();

	if (returnCode < 0)
		return returnCode;

	std::vector<BatchElement> batchTail;
	for (int i = returnCode + 1; i < currentBatchSz; ++i)
		batchTail.push_back(currentBatch.at(i));
	currentBatch = batchTail;

	return returnCode;
}

static int fillVideoFrameBatch(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch
) {
	ChronoTimer timer;
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
	timer.printLastPointDelta("MS for batch's filling: ", logStreams.timeStream);
	return skippedFrames;
}


static int findGoodFrameFromBatch(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	std::vector<bool> &isMatchesEstimated,
	std::vector<std::vector<DMatch>> &estimatedMatches,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
) {
	int batchSz = currentBatch.size();
	Mat goodFrame;
	std::vector<KeyPoint> goodFeatures;
	std::vector<DMatch> goodMatches;
	int goodIndex = FRAME_NOT_FOUND;
	Mat candidateFrame;
	std::vector<KeyPoint> candidateFrameFeatures;
	std::vector<DMatch> candidateMatches;
	for (
		int batchIndex = batchSz - 1;
		batchIndex >= dataProcessingConditions.skipFramesFromBatchHead;
		batchIndex--
	) {
		while (!isMatchesEstimated.at(batchIndex));
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
			if (dataProcessingConditions.useFirstFitInBatch)
				break;
		}
	}
	if (goodIndex != FRAME_NOT_FOUND) {
		newGoodFrame = goodFrame.clone();
		newFeatures = goodFeatures;
		matches = goodMatches;
		return goodIndex;
	}

	return goodIndex;
}