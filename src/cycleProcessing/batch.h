#pragma once

#include "mainCycleStructures.h"

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
int findGoodFrameFromBatch(
	MediaSources &mediaInputStruct,
	const DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame, Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
);