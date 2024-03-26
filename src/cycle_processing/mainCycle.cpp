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


void mainCycle(std::deque<TemporalImageData> &temporalImageDataDeque, GlobalData &globalDataStruct) {
    MediaSources mediaInputStruct;
    DataProcessingConditions dataProcessingConditions;
    defineProcessingEnvironment(mediaInputStruct, dataProcessingConditions);

    Mat lastGoodFrame;
    if (!processingFirstPairFrames(mediaInputStruct, dataProcessingConditions,
            temporalImageDataDeque, lastGoodFrame, globalDataStruct.spatialPoints,
            globalDataStruct.spatialPointsColors)
    ) {
        // Error message if at least two good frames are not found in the video
        std::cerr << "Couldn't find at least two good frames in fitst video batch" << std::endl;
        exit(-1);
    }
    globalDataStruct.cameraRotations.push_back(temporalImageDataDeque.at(1).rotation);

    int lastFrameIdx = 1;
    while (true) {
		logStreams.mainReportStream << std::endl << "================================================================\n" << std::endl;

        // Find the next good frame batch
        Mat nextGoodFrame;
        int hasVideoGoodFrames = findGoodFrameFromBatch(mediaInputStruct,
                                    dataProcessingConditions, lastGoodFrame,
                                    temporalImageDataDeque.at(lastFrameIdx).allExtractedFeatures,
                                    nextGoodFrame,
                                    temporalImageDataDeque.at(lastFrameIdx+1).allExtractedFeatures,
                                    temporalImageDataDeque.at(lastFrameIdx+1).allMatches);
        if (hasVideoGoodFrames == 0) {
            logStreams.mainReportStream << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
            logStreams.mainReportStream << "Video is over. No more frames..." << std::endl;
        } else if (hasVideoGoodFrames < 0) {
            std::cerr << "No good frames in batch. Stop video processing" << std::endl;
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
            oldSpatialPointsForNewFrame, newFrameFeatureCoords);

        // Find transformation matrix
		if (oldSpatialPointsForNewFrame.size() != newFrameFeatureCoords.size() // А такое вообще может быть???
            || oldSpatialPointsForNewFrame.size() < 4
        ) {
			logStreams.mainReportStream << "Not enough corresponding points for solvePnP RANSAC" << std::endl;
			break;
		}
        Mat rotationVector;
        solvePnPRansac(oldSpatialPointsForNewFrame, newFrameFeatureCoords,
            dataProcessingConditions.calibrationMatrix, noArray(), rotationVector,
            temporalImageDataDeque.at(lastFrameIdx+1).motion);
        // Convert rotation vector to rotation matrix
        Rodrigues(rotationVector, temporalImageDataDeque.at(lastFrameIdx+1).rotation);
        globalDataStruct.cameraRotations.push_back(
            temporalImageDataDeque.at(lastFrameIdx+1).rotation);

		logStreams.mainReportStream << "Used in solvePnP: " << oldSpatialPointsForNewFrame.size() << std::endl;
		logStreams.mainReportStream << temporalImageDataDeque.at(lastFrameIdx+1).rotation << std::endl;
		logStreams.mainReportStream << temporalImageDataDeque.at(lastFrameIdx+1).motion.t() << std::endl;
		logStreams.mainReportStream.flush();
		rawOutput(temporalImageDataDeque.at(lastFrameIdx+1).motion.t(), logStreams.poseStream);
		logStreams.poseStream.flush();

        // Get matched point coordinates for reconstruction
        std::vector<Point2f> matchedPointCoords1;
        std::vector<Point2f> matchedPointCoords2;
        std::vector<Point3f> newSpatialPoints;
        getKeyPointCoordsFromFramePair(
            temporalImageDataDeque.at(lastFrameIdx).allExtractedFeatures,
            temporalImageDataDeque.at(lastFrameIdx+1).allExtractedFeatures,
            temporalImageDataDeque.at(lastFrameIdx+1).allMatches,
            matchedPointCoords1, matchedPointCoords2);
        reconstruct(dataProcessingConditions.calibrationMatrix,
            temporalImageDataDeque.at(lastFrameIdx).rotation,
            temporalImageDataDeque.at(lastFrameIdx).motion,
            temporalImageDataDeque.at(lastFrameIdx+1).rotation,
            temporalImageDataDeque.at(lastFrameIdx+1).motion,
            matchedPointCoords1, matchedPointCoords2, newSpatialPoints);
		pushNewSpatialPoints(nextGoodFrame, newSpatialPoints, globalDataStruct,
            temporalImageDataDeque.at(lastFrameIdx).correspondSpatialPointIdx,
            temporalImageDataDeque.at(lastFrameIdx+1));

        // Update last good frame
        lastGoodFrame = nextGoodFrame.clone();
        if (lastFrameIdx == OPTIMAL_DEQUE_SIZE - 2) {
            temporalImageDataDeque.pop_front();
			temporalImageDataDeque.push_back({});
        } else {
            lastFrameIdx++;
        }
    }

	rawOutput(globalDataStruct.spatialPoints, logStreams.pointsStream);
	logStreams.pointsStream.flush();
}


static bool processingFirstPairFrames(
    MediaSources &mediaInputStruct, const DataProcessingConditions &dataProcessingConditions,
    std::deque<TemporalImageData> &temporalImageDataDeque, Mat &secondFrame,
    std::vector<Point3f> &spatialPoints, std::vector<cv::Vec3b> &firstPairSpatialPointColors)
{
    Mat firstFrame;
    if (!findFirstGoodFrame(mediaInputStruct, dataProcessingConditions,firstFrame, 
            temporalImageDataDeque.at(0).allExtractedFeatures)
    ) {
        logStreams.mainReportStream << std::endl << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        logStreams.mainReportStream << "Video is over. No more frames..." << std::endl;
        return false;
    }

    int videoGoodFramesCnt = findGoodFrameFromBatch(mediaInputStruct, 
                                dataProcessingConditions, firstFrame,
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
    computeTransformationAndFilterPoints(dataProcessingConditions,
        temporalImageDataDeque.at(0),temporalImageDataDeque.at(1),
        extractedPointCoords1, extractedPointCoords2, chiralityMask);
    reconstruct(dataProcessingConditions.calibrationMatrix,
        temporalImageDataDeque.at(0).rotation, temporalImageDataDeque.at(0).motion,
        temporalImageDataDeque.at(1).rotation, temporalImageDataDeque.at(1).motion,
        extractedPointCoords1, extractedPointCoords2, spatialPoints);
    defineFeaturesCorrespondSpatialIndices(chiralityMask, secondFrame, temporalImageDataDeque.at(0),
        temporalImageDataDeque.at(1), firstPairSpatialPointColors);

    return true;
}


static int findGoodFrameFromBatch(
    MediaSources &mediaInputStruct, const DataProcessingConditions &dataProcessingConditions,
    Mat &previousFrame, std::vector<KeyPoint> &previousFeatures, Mat &newGoodFrame,
    std::vector<KeyPoint> &newFeatures, std::vector<DMatch> &matches)
{
    std::vector<Mat> frameBatch;
	std::vector<std::vector<KeyPoint>> batchFeatures;
    fillVideoFrameBatch(mediaInputStruct, dataProcessingConditions, frameBatch, batchFeatures);
    
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
                                dataProcessingConditions.matcherType, matches);

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


static void fillVideoFrameBatch(
    MediaSources &mediaInputStruct, const DataProcessingConditions &dataProcessingConditions,
    std::vector<Mat> &frameBatch, std::vector<std::vector<KeyPoint>> &batchFeatures)
{
    logStreams.mainReportStream << "Features count in frames added to batch: ";
    Mat nextFrame;
	std::vector<KeyPoint> nextFeatures;
	int skippedFrames = 0;
    while (frameBatch.size() < dataProcessingConditions.frameBatchSize 
           && getNextFrame(mediaInputStruct, nextFrame)
    ) {
        /// TODO: In future we can do UNDISTORTION here
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
