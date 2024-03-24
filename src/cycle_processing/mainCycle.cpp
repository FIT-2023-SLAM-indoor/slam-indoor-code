#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../cameraCalibration.h"
#include "../cameraTransition.h"
#include "../fastExtractor.h"
#include "../featureMatching.h"
#include "../triangulate.h"
#include "../IOmisc.h"

#include "../config/config.h"

#include "mainCycle.h"
#include "mainCycleInternals.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;


void mainCycle(GlobalData globalDataStruct) {
    //// TODO: Объединить в одну функцию
    MediaSources mediaInputStruct;
    DataProcessingConditions dataProcessingConditions;
    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
    defineMediaSources(mediaInputStruct);
    defineProcessingConditions(dataProcessingConditions);
    initTemporalImageDataDeque(temporalImageDataDeque);
    ////

    Mat lastGoodFrame;
    // Process the first pair of frames
    if (!processingFirstPairFrames(mediaInputStruct, dataProcessingConditions,
            temporalImageDataDeque, lastGoodFrame, globalDataStruct.spatialPoints)
    ) {
        // Error message if at least two good frames are not found in the video
        std::cerr << "Couldn't find at least two good frames in fitst video batch" << std::endl;
        exit(-1);
    }

    int lastGoodFrameIdx = 1;
    Mat nextGoodFrame;
    bool hasVideoGoodFrames;
    while (true) {
		logStreams.mainReportStream << std::endl << "================================================================\n" << std::endl;

        // Find the next good frame batch
        hasVideoGoodFrames = findGoodVideoFrameFromBatch(mediaInputStruct,
                                dataProcessingConditions, lastGoodFrame,
                                temporalImageDataDeque.at(lastGoodFrameIdx).allExtractedFeatures,
                                nextGoodFrame,
                                temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
                                temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches);
        if (hasVideoGoodFrames == 0) {
            logStreams.mainReportStream << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
            logStreams.mainReportStream << "Video is over. No more frames..." << std::endl;
        } else if (hasVideoGoodFrames < 0) {
            std::cerr << "No good frames in batch. Stop video processing" << std::endl;
            break;
        }

        // Get object and image points for reconstruction
        std::vector<Point3f> objPoints;
        std::vector<Point2f> imgPoints;
        getObjAndImgPoints(temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches,
            temporalImageDataDeque.at(lastGoodFrameIdx).correspondSpatialPointIdx,
            globalDataStruct.spatialPoints,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
            objPoints, imgPoints);

        // Find transformation matrix
		if (objPoints.size() != imgPoints.size() || objPoints.size() < 4) {
			logStreams.mainReportStream << "Not enough corresponding points for solvePnP RANSAC" << std::endl;
			break;
		}
        Mat rotationVector;
        solvePnPRansac(objPoints, imgPoints, dataProcessingConditions.calibrationMatrix,
            noArray(), rotationVector, temporalImageDataDeque.at(lastGoodFrameIdx+1).motion);

        // Convert rotation vector to rotation matrix
        Rodrigues(rotationVector, temporalImageDataDeque.at(lastGoodFrameIdx+1).rotation);

		logStreams.mainReportStream << "Used in solvePnP: " << objPoints.size() << std::endl;
		logStreams.mainReportStream << temporalImageDataDeque.at(lastGoodFrameIdx+1).rotation << std::endl;
		logStreams.mainReportStream << temporalImageDataDeque.at(lastGoodFrameIdx+1).motion.t() << std::endl;
		logStreams.mainReportStream.flush();
		rawOutput(temporalImageDataDeque.at(lastGoodFrameIdx+1).motion.t(), logStreams.poseStream);
		logStreams.poseStream.flush();

        // Get matched point coordinates for reconstruction
        std::vector<Point2f> matchedPointCoords1;
        std::vector<Point2f> matchedPointCoords2;
        std::vector<Point3f> newSpatialPoints;
        getMatchedPointCoords(temporalImageDataDeque.at(lastGoodFrameIdx).allExtractedFeatures,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches,
            matchedPointCoords1, matchedPointCoords2);

        // Reconstruct 3D spatial points
        reconstruct(dataProcessingConditions.calibrationMatrix,
            temporalImageDataDeque.at(lastGoodFrameIdx).rotation,
            temporalImageDataDeque.at(lastGoodFrameIdx).motion,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).rotation,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).motion, 
            matchedPointCoords1, matchedPointCoords2, newSpatialPoints);

		temporalImageDataDeque.at(lastGoodFrameIdx+1).correspondSpatialPointIdx.resize(
				temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures.size(), -1
		);
		pushNewSpatialPoints(temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches,
							 temporalImageDataDeque.at(lastGoodFrameIdx).correspondSpatialPointIdx,
							 temporalImageDataDeque.at(lastGoodFrameIdx+1).correspondSpatialPointIdx,
							 newSpatialPoints, globalDataStruct.spatialPoints);

        // Update last good frame
        lastGoodFrame = nextGoodFrame.clone();  // будет ли в будущем освобождаться от старых значений nextGoodFrame?
        if (lastGoodFrameIdx == OPTIMAL_DEQUE_SIZE - 2) {
            temporalImageDataDeque.pop_front();
			temporalImageDataDeque.push_back({});
        } else {
            lastGoodFrameIdx++;
        }
    }

	rawOutput(globalDataStruct.spatialPoints, logStreams.pointsStream);
	logStreams.pointsStream.flush();
}


static bool findFirstGoodVideoFrameAndFeatures(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (getNextFrame(mediaInputStruct, candidateFrame)) {
        // TODO: In future we can do UNDISTORTION here
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


static void fillVideoFrameBatch(
    MediaSources &mediaInputStruct, DataProcessingConditions &dataProcessingConditions,
    int frameBatchSize, std::vector<Mat> &frameBatch, 
    std::vector<std::vector<KeyPoint>> &batchFeatures)
{
    logStreams.mainReportStream << "Features count in frames added to batch: ";
    Mat nextFrame;
	std::vector<KeyPoint> nextFeatures;
	int skippedFrames = 0;
    while (frameBatch.size() < frameBatchSize && getNextFrame(mediaInputStruct, nextFrame)) {
        // TODO: In future we can do UNDISTORTION here
		fastExtractor(nextFrame, nextFeatures, 
            dataProcessingConditions.featureExtractingThreshold);
		if (nextFeatures.size() >= dataProcessingConditions.requiredExtractedPointsCount) {
            logStreams.mainReportStream << nextFeatures.size() << " ";
            frameBatch.push_back(nextFrame.clone());
            batchFeatures.push_back(nextFeatures);
		} else {
            skippedFrames++;
        }
    }

    logStreams.mainReportStream << std::endl << "Skipped frames while constructing batch: ";
    logStreams.mainReportStream << skippedFrames << std::endl;
}


static int findGoodVideoFrameFromBatch(
    MediaSources &mediaInputStruct, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    Mat &previousFrame, std::vector<KeyPoint> &previousFeatures, 
    Mat &newGoodFrame, std::vector<KeyPoint> &newFeatures,
    std::vector<DMatch> &matches)
{
    std::vector<Mat> frameBatch;
	std::vector<std::vector<KeyPoint>> batchFeatures;
    fillVideoFrameBatch(mediaInputStruct, dataProcessingConditions, 
        frameBatchSize, frameBatch, batchFeatures);
    
    if (frameBatch.size() == 0) {
        return 0;
    }

	logStreams.mainReportStream << "Prev. extracted: " << previousFeatures.size() << std::endl;
    Mat candidateFrame;
	std::vector<KeyPoint> candidateFrameFeatures;
    for (int frameIndex = frameBatch.size() - 1; frameIndex >= 0; frameIndex--) {
		candidateFrame = frameBatch.at(frameIndex).clone();
		candidateFrameFeatures = batchFeatures.at(frameIndex);

		matches.clear();
        // Match features between the previous frame and the new frame
        matchFramesPairFeatures(previousFrame, candidateFrame,
                                previousFeatures, candidateFrameFeatures,
                                dataProcessingConditions, matches);

		logStreams.mainReportStream << "Batch index: " << frameIndex
									<< "; curr. extracted: " << candidateFrameFeatures.size()
									<< "; matched " << matches.size() << std::endl;
        // Check if enough matches are found
        if (matches.size() >= dataProcessingConditions.requiredMatchedPointsCount) {
			newGoodFrame = candidateFrame.clone();
			newFeatures.resize(candidateFrameFeatures.size());
			std::copy(candidateFrameFeatures.begin(), candidateFrameFeatures.end(), newFeatures.begin());
            return frameBatch.size();
        }
    }

    // No good frame found in the batch
    return -1;
}


static bool processingFirstPairFrames(
    MediaSources &mediaInputStruct, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    std::deque<TemporalImageData> &temporalImageDataDeque,
    Mat &secondFrame, std::vector<Point3f> &spatialPoints)
{
    Mat firstFrame;
    if (!findFirstGoodVideoFrameAndFeatures(mediaInputStruct, dataProcessingConditions,
            firstFrame, temporalImageDataDeque.at(0).allExtractedFeatures)
    ) {
        logStreams.mainReportStream << std::endl << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        logStreams.mainReportStream << "Video is over. No more frames..." << std::endl;
        return false;
    }
    defineInitialCameraPosition(temporalImageDataDeque.at(0));


    int videoGoodFramesCnt = findGoodVideoFrameFromBatch(mediaInputStruct, frameBatchSize,
                            dataProcessingConditions,firstFrame,
                            temporalImageDataDeque.at(0).allExtractedFeatures,
                            secondFrame,temporalImageDataDeque.at(1).allExtractedFeatures,
                            temporalImageDataDeque.at(1).allMatches);
    if (videoGoodFramesCnt == 0) {
        logStreams.mainReportStream << std::endl << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        logStreams.mainReportStream << "Video is over. No more frames..." << std::endl;
        return false;
    } else if (videoGoodFramesCnt < 0) {
        return false;
    }

    Mat chiralityMask;
    std::vector<Point2f> extractedPointCoords1, extractedPointCoords2;
    computeTransformationAndMaskPoints(dataProcessingConditions, chiralityMask,
        temporalImageDataDeque.at(0),temporalImageDataDeque.at(1),
        extractedPointCoords1, extractedPointCoords2);
    reconstruct(dataProcessingConditions.calibrationMatrix,
        temporalImageDataDeque.at(0).rotation, temporalImageDataDeque.at(0).motion,
        temporalImageDataDeque.at(1).rotation, temporalImageDataDeque.at(1).motion,
        extractedPointCoords1, extractedPointCoords2, spatialPoints);
    defineCorrespondenceIndices(dataProcessingConditions, chiralityMask,
        temporalImageDataDeque.at(0), temporalImageDataDeque.at(1));

    return true;
}


static void pushNewSpatialPoints(
	const std::vector<DMatch> &matches,
	std::vector<int> &prevFrameCorrespondingIndices,
	std::vector<int> &currFrameCorrespondingIndices,
	const std::vector<Point3f> &newSpatialPoints,
	std::vector<Point3f> &allSpatialPoints
) {
	for (int i = 0; i < matches.size(); ++i)
	{
		int queryIdx = matches[i].queryIdx;
		int trainIdx = matches[i].trainIdx;

		int structIdx = prevFrameCorrespondingIndices[queryIdx];
		if (structIdx >= 0) //If the point already exists in space, the space points corresponding to the pair of matching points should be the same, with the same index
		{
			currFrameCorrespondingIndices[trainIdx] = structIdx;
			continue;
		}

		//If the point already exists in space, add the point to the structure, and the spatial point indexes of the pair of matching points are the indexes of the newly added points
		allSpatialPoints.push_back(newSpatialPoints[i]);
		prevFrameCorrespondingIndices[queryIdx] = ((int)allSpatialPoints.size()) - 1;
		currFrameCorrespondingIndices[trainIdx] = ((int)allSpatialPoints.size()) - 1;
	}
}
