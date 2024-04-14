#pragma once

#include "mainCycleStructures.h"

#define EMPTY_BATCH (-2)
#define FRAME_NOT_FOUND (-1)

/**
 * TODO: ADD DOCS
 *
 * @param mediaInputStruct
 * @param dataProcessingConditions
 * @param currentBatch
 * @param previousFrame
 * @param newGoodFrame
 * @param previousFeatures
 * @param newFeatures
 * @param matches
 * @return
 */
int findGoodFrameFromBatchMultithreadingWrapper(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);