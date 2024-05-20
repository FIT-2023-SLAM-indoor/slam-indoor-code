#include "thread"

#include "../misc/ChronoTimer.h"
#include "../IOmisc.h"
#include "../fastExtractor.h"

#include "../featureMatching/featureMatching.h"

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

static int findGoodFrameFromMatchedBatch(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);

static int findGoodFramesFromBatchSingleThread(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);

static int findGoodFramesFromBatchMultiThreads(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);

int findGoodFrameFromBatch(
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
	if (currentBatch.size() == 0)
		return EMPTY_BATCH;
	logStreams.mainReportStream << "Prev. extracted: " << previousFeatures.size() << std::endl;

	int goodIndex;
	if (dataProcessingConditions.threadsCount <= 1) {
		goodIndex = findGoodFramesFromBatchSingleThread(
			dataProcessingConditions, currentBatch,
			previousFrame, newGoodFrame,
			previousFeatures, newFeatures,
			matches
		);
	}
	else {
		goodIndex = findGoodFramesFromBatchMultiThreads(
			dataProcessingConditions, currentBatch,
			previousFrame, newGoodFrame,
			previousFeatures, newFeatures,
			matches
		);
	}

	if (goodIndex >= 0) {
		std::vector<BatchElement> batchTail;
		for (int i = goodIndex + 1; i < currentBatchSz; ++i)
			batchTail.push_back(currentBatch.at(i));
		currentBatch = batchTail;
	}
	return goodIndex;
}

static int findGoodFramesFromBatchSingleThread(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
) {
	int currentBatchSz = currentBatch.size();
	ChronoTimer timer;

	Mat previousDescriptor;
	extractDescriptor(previousFrame, previousFeatures,
		dataProcessingConditions.matcherType, previousDescriptor);

	Mat goodFrame;
	std::vector<KeyPoint> goodFeatures;
	std::vector<DMatch> goodMatches;
	int goodIndex = FRAME_NOT_FOUND;
	for (
		int batchIndex = currentBatchSz - 1;
		batchIndex >= dataProcessingConditions.skipFramesFromBatchHead;
		batchIndex--
	) {
		BatchElement &element = currentBatch.at(batchIndex);

		matchFramesPairFeatures(
			previousDescriptor, element.frame, element.features,
			dataProcessingConditions.matcherType, element.matches
		);

		logStreams.mainReportStream << "Batch index: " << batchIndex
									<< "; curr. extracted: " << element.features.size()
									<< "; matched " << element.matches.size() << std::endl;

		if (
			element.matches.size() >= dataProcessingConditions.requiredMatchedPointsCount
			&& element.matches.size() >= goodMatches.size()
		) {
			logStreams.mainReportStream << "Frame " << batchIndex << " is a good" << std::endl;
			goodIndex = batchIndex;
			goodFrame = element.frame.clone();
			goodFeatures = element.features;
			goodMatches = element.matches;
			if (dataProcessingConditions.useFirstFitInBatch)
				break;
		}
	}
	if (goodIndex != FRAME_NOT_FOUND) {
		newGoodFrame = goodFrame.clone();
		newFeatures = goodFeatures;
		matches = goodMatches;
	}

	logStreams.timeStream << "Matching time for index " << goodIndex;
	timer.printLastPointDelta(": ", logStreams.timeStream);
	timer.updateLastPoint();

	return goodIndex;
}

static int findGoodFramesFromBatchMultiThreads(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
) {
	int currentBatchSz = currentBatch.size();
	ChronoTimer timer;

	int threadsCnt = dataProcessingConditions.threadsCount;
	std::vector<std::thread> threads;
	volatile bool threadsShouldDie = false;

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
				matchFramesPairFeatures(
					previousDescriptor, element.frame, element.features,
					dataProcessingConditions.matcherType, element.matches
				);
				element.estimated = true;
				std::cout << "Finish: " << batchIndex << "; matched: " << element.matches.size() << std::endl;
				this_thread::yield();
				if (threadsShouldDie)
					break;
			}
		}, i));
	}

	timer.printLastPointDelta("Matching threads started at: ", logStreams.timeStream);
	timer.updateLastPoint();

	int goodIndex = findGoodFrameFromMatchedBatch(
		dataProcessingConditions,
		currentBatch,
		previousFrame, newGoodFrame,
		previousFeatures, newFeatures,
		matches
	);

	logStreams.timeStream << "Matching time for index " << goodIndex;
	timer.printLastPointDelta(": ", logStreams.timeStream);
	timer.updateLastPoint();

	threadsShouldDie = true;
	for (auto& thread : threads)
		thread.join();

	timer.printLastPointDelta("Threads terminated: ", logStreams.timeStream);
	timer.updateLastPoint();

	return goodIndex;
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
			nextFeatures,
			std::vector<DMatch>(),
			false
		});
		currentFrameBatchSize++;
	}
	logStreams.mainReportStream << std::endl << "Skipped for first: " << skippedFramesForFirstFound << std::endl;
	logStreams.mainReportStream << "Skipped frames while constructing batch: " << skippedFrames << std::endl;
	logStreams.mainReportStream << "Batch size: " << currentBatch.size() << std::endl;
	timer.printLastPointDelta("MS for batch's filling: ", logStreams.timeStream);
	return skippedFrames;
}


static int findGoodFrameFromMatchedBatch(
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
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
	for (
		int batchIndex = batchSz - 1;
		batchIndex >= dataProcessingConditions.skipFramesFromBatchHead;
		batchIndex--
	) {
		while (!currentBatch.at(batchIndex).estimated);
		BatchElement &element = currentBatch.at(batchIndex);

		logStreams.mainReportStream << "Batch index: " << batchIndex
									<< "; curr. extracted: " << element.features.size()
									<< "; matched " << element.matches.size() << std::endl;

		if (
			element.matches.size() >= dataProcessingConditions.requiredMatchedPointsCount
			&& element.matches.size() >= goodMatches.size()
		) {
			logStreams.mainReportStream << "Frame " << batchIndex << " is a good" << std::endl;
			goodIndex = batchIndex;
			goodFrame = element.frame.clone();
			goodFeatures = element.features;
			goodMatches = element.matches;
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