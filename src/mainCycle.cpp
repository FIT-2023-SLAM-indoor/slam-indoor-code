#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ceres/ceres.h>
#include <opencv2/sfm.hpp>

#include "triangulate.h"
#include "vizualizationModule.h"
#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "fastExtractor.h"
#include "featureMatching.h"
#include "IOmisc.h"
#include "bundleAdjustment.h"

#include "config/config.h"

#include "mainCycle.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;

/*
Список того, на что я пока решил забить:
1) Сохранение цветов фич - надо немного переписать Extractor (ну или добавить это в функции поиска frame-ов)
2) Выбор применения Undistortion-а к кадрам - сейчас он просто применяется
3) Обработка крайних случаев: этот код нужно хорошо отревьюить, я толком не думал про небезопасные места
*/


void defineCalibrationMatrix(Mat &calibrationMatrix) {
    // Сreating a new matrix or changing the type and size of an existing one
    calibrationMatrix.create(3, 3, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);
}


void defineDistortionCoeffs(Mat &distortionCoeffs) {
    // Сreating a new matrix or changing the type and size of an existing one
    distortionCoeffs.create(1, 5, CV_64F);
    loadMatrixFromXML(
		configService.getValue<std::string>(ConfigFieldEnum::CALIBRATION_PATH_).c_str(),
		distortionCoeffs, "DC"
	);
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
void defineMediaSources(MediaSources &mediaInputStruct) {
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
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

void defineProcessingConditions(
	int batchSize,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType,
    DataProcessingConditions &dataProcessingConditions)
{
    defineCalibrationMatrix(dataProcessingConditions.calibrationMatrix);
    defineDistortionCoeffs(dataProcessingConditions.distortionCoeffs);

	dataProcessingConditions.batchSize = batchSize;
    dataProcessingConditions.featureExtractingThreshold = featureExtractingThreshold;
    dataProcessingConditions.requiredExtractedPointsCount = requiredExtractedPointsCount;
    dataProcessingConditions.requiredMatchedPointsCount = requiredMatchedPointsCount;
    dataProcessingConditions.matcherType = matcherType;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool getNextFrame(MediaSources &mediaInputStruct, Mat& nextFrame) {
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
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

bool findFirstGoodVideoFrameAndFeatures(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (getNextFrame(mediaInputStruct, candidateFrame)) {
//        undistort(candidateFrame, goodFrame,
//            dataProcessingConditions.calibrationMatrix,
//            dataProcessingConditions.distortionCoeffs);
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


void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    DataProcessingConditions &dataProcessingConditions,
    std::vector<DMatch>& matches)
{
    // Extract descriptors from the key points of the input frames
    Mat firstDescriptor;
    extractDescriptor(firstFrame, firstFeatures, 
        dataProcessingConditions.matcherType, firstDescriptor);
    Mat secondDescriptor;
    extractDescriptor(secondFrame, secondFeatures, 
        dataProcessingConditions.matcherType, secondDescriptor);

    // Match the descriptors using the specified matcher type
    matchFeatures(firstDescriptor, secondDescriptor, matches, 
        dataProcessingConditions.matcherType);
}

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
	DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch
) {
    int currentFrameBatchSize = 0;
    Mat nextFrame;

	std::vector<KeyPoint> nextFeatures;
	int skippedFrames = 0, skippedFramesForFirstFound = 0;
	logStreams.mainReportStream << "Features count in frames added to batch: ";
    while (
		currentBatch.size() < dataProcessingConditions.batchSize
		&& getNextFrame(mediaInputStruct, nextFrame)
	) {
		// Учитывая новый вариант сбора батча, undistort тоже надо делать тут, если всё же будем им заниматься
//        undistort(candidateFrame, newGoodFrame,
//                  dataProcessingConditions.calibrationMatrix,
//                  dataProcessingConditions.distortionCoeffs);
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
	return skippedFrames;
}

/**
 * Find new frame from batch which can be up to dataProcessingConditions.batchSize
 *
 * @param [in] mediaInputStruct
 * @param [in] dataProcessingConditions
 * @param [in,out] currentBatch <ul>
 * 		<li>Input is a batch tail from previous iteration</li>
 * 		<li>Input is a batch tail after new good frame was found</li>
 * </ul>
 * @param [in] previousFrame
 * @param [out] newGoodFrame
 * @param [in] previousFeatures
 * @param [out] newFeatures
 * @param [out] matches
 * @return true when new good frame was found
 */
static bool findGoodVideoFrameFromBatch(
	MediaSources &mediaInputStruct,
	DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
	Mat &previousFrame,
	Mat &newGoodFrame,
	std::vector<KeyPoint> &previousFeatures,
	std::vector<KeyPoint> &newFeatures,
	std::vector<DMatch> &matches
) {
	int skippedFramesCount = fillVideoFrameBatch(mediaInputStruct, dataProcessingConditions, currentBatch);
	int currentBatchSz = currentBatch.size();

	logStreams.mainReportStream << "Prev. extracted: " << previousFeatures.size() << std::endl;
	Mat goodFrame;
	std::vector<KeyPoint> goodFeatures;
	std::vector<DMatch> goodMatches;
	int goodIndex = -1;
	Mat candidateFrame;
	std::vector<KeyPoint> candidateFrameFeatures;
	std::vector<DMatch> candidateMatches;
	for (int batchIndex = currentBatchSz - 1; batchIndex >= 0; batchIndex--) {
		candidateFrame = currentBatch.at(batchIndex).frame.clone();
		candidateFrameFeatures = currentBatch.at(batchIndex).features;

		matchFramesPairFeatures(previousFrame, candidateFrame,
								previousFeatures, candidateFrameFeatures,
								dataProcessingConditions, candidateMatches);

		logStreams.mainReportStream << "Batch index: " << batchIndex
									<< "; curr. extracted: " << candidateFrameFeatures.size()
									<< "; matched " << candidateMatches.size() << std::endl;

		if (
			candidateMatches.size() >= dataProcessingConditions.requiredMatchedPointsCount
			&& candidateMatches.size() >= goodMatches.size()
		) {
			goodIndex = batchIndex;
			goodFrame = candidateFrame.clone();
			goodFeatures = candidateFrameFeatures;
			goodMatches = candidateMatches;
			break;
		}
	}
	if (goodIndex != -1) {
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
		return true;
	}

	// No good frame found in the batch
	return false;
}

void defineInitialCameraPosition(TemporalImageData &initialFrame) {
    initialFrame.rotation = Mat::eye(3, 3, CV_64FC1);
    initialFrame.motion = Mat::zeros(3, 1, CV_64FC1);
}


void maskoutPoints(const Mat &chiralityMask, std::vector<Point2f> &extractedPoints) {
    // Ensure that the sizes of the chirality mask and points vector match
    if (chiralityMask.rows != extractedPoints.size()) {
        std::cerr << "Chirality mask size does not match points vector size" << std::endl;
        exit(-1);
    }

    std::vector<Point2f> pointsCopy = extractedPoints;
    extractedPoints.clear();

    for (int i = 0; i < chiralityMask.rows; i++) {
        if (chiralityMask.at<uchar>(i) > 0) {
            extractedPoints.push_back(pointsCopy[i]);
        }
    }
}


void computeTransformationAndMaskPoints(
    DataProcessingConditions &dataProcessingConditions, Mat &chiralityMask,
    TemporalImageData &prevFrameData, TemporalImageData &newFrameData,
    std::vector<Point2f> &extractedPointCoords1, std::vector<Point2f> &extractedPointCoords2)
{
	getMatchedPointCoords(prevFrameData.allExtractedFeatures, newFrameData.allExtractedFeatures,
						  newFrameData.allMatches, extractedPointCoords1, extractedPointCoords2);

    // Насколько я понял для вычисления матрицы вращения, вектора смещения - мне нужна эта функция
    estimateTransformation(
        extractedPointCoords1, extractedPointCoords2, dataProcessingConditions.calibrationMatrix,
        newFrameData.rotation, newFrameData.motion, chiralityMask);

    // Apply the chirality mask to the points from the frames
    maskoutPoints(chiralityMask, extractedPointCoords1);
    maskoutPoints(chiralityMask, extractedPointCoords2);
}


void defineCorrespondenceIndices(
    DataProcessingConditions &dataProcessingConditions, Mat &chiralityMask,
    TemporalImageData &prevFrameData, TemporalImageData &newFrameData)
{
    // Resize the correspondence spatial point indices vectors for the previous and new frames
    prevFrameData.correspondSpatialPointIdx.resize(
        prevFrameData.allExtractedFeatures.size(), -1);
    newFrameData.correspondSpatialPointIdx.resize(
        newFrameData.allExtractedFeatures.size(), -1);

    int newMatchIdx = 0;
    for (int matchIdx = 0; matchIdx < newFrameData.allMatches.size(); matchIdx++) {
        // Check if the match is valid based on the chirality mask
        if (chiralityMask.at<uchar>(matchIdx) > 0) {
            // Update correspondence indices for keypoints in the previous and new frames
            prevFrameData.correspondSpatialPointIdx.at(
                newFrameData.allMatches[matchIdx].queryIdx) = newMatchIdx;
            newFrameData.correspondSpatialPointIdx.at(
                newFrameData.allMatches[matchIdx].trainIdx) = newMatchIdx;
            newMatchIdx++;
        }
    }
}


bool processingFirstPairFrames(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions,
	std::vector<BatchElement> &currentBatch,
    std::deque<TemporalImageData> &temporalImageDataDeque,
    Mat &secondFrame, std::vector<Point3f> &spatialPoints
) {
    Mat firstFrame;
    if (!findFirstGoodVideoFrameAndFeatures(mediaInputStruct, dataProcessingConditions,
            firstFrame, temporalImageDataDeque.at(0).allExtractedFeatures)
    ) {
        return false;
    }
    defineInitialCameraPosition(temporalImageDataDeque.at(0));

    if (!findGoodVideoFrameFromBatch(
		mediaInputStruct, dataProcessingConditions, currentBatch,
		firstFrame, secondFrame,
		temporalImageDataDeque.at(0).allExtractedFeatures,
		temporalImageDataDeque.at(1).allExtractedFeatures,
		temporalImageDataDeque.at(1).allMatches
	)) {
        return false;
    }
    
    // Из докстрингов хэдеров OpenCV я не понял нужно ли мне задавать размерность этой матрице
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


void getObjAndImgPoints(
    std::vector<DMatch> &matches,
    std::vector<int> &correspondSpatialPointIdx,
    std::vector<Point3f> &spatialPoints,
    std::vector<KeyPoint> &extractedFeatures,
    std::vector<Point3f> &objPoints,
    std::vector<Point2f> &imgPoints)
{
    for (auto &match : matches) {
        int query_idx = match.queryIdx;
        int train_idx = match.trainIdx;

        int struct_idx = correspondSpatialPointIdx[query_idx];
        if (struct_idx >= 0) {
            objPoints.push_back(spatialPoints[struct_idx]);
            imgPoints.push_back(extractedFeatures[train_idx].pt);
        }
    }
}


void getMatchedPointCoords(
	std::vector<KeyPoint> &firstExtractedFeatures,
	std::vector<KeyPoint> &secondExtractedFeatures,
	std::vector<DMatch> &matches,
	std::vector<Point2f> &firstMatchedPoints,
	std::vector<Point2f> &secondMatchedPoints)
{
	firstMatchedPoints.clear();
	secondMatchedPoints.clear();
	for (int i = 0; i < matches.size(); i++) {
		firstMatchedPoints.push_back(firstExtractedFeatures[matches[i].queryIdx].pt);
		secondMatchedPoints.push_back(secondExtractedFeatures[matches[i].trainIdx].pt);
	}
}


void initTemporalImageDataDeque(std::deque<TemporalImageData> &temporalImageDataDeque) {
    for (int imageDataIdx = 0; imageDataIdx < temporalImageDataDeque.size(); imageDataIdx++) {
        temporalImageDataDeque.at(imageDataIdx).allExtractedFeatures.clear();
        temporalImageDataDeque.at(imageDataIdx).allMatches.clear();
        temporalImageDataDeque.at(imageDataIdx).correspondSpatialPointIdx.clear();
    }
}

/**
 * Merges new spatial points and marks them in corresponding indices tables.
 *
 * @param [in] matches
 * @param [in, out] prevFrameCorrespondingIndices
 * @param [out] currFrameCorrespondingIndices
 * @param [in] newSpatialPoints
 * @param [out] allSpatialPoints
 */
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

void mainCycle(
    int frameBatchSize,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType
) {
    MediaSources mediaInputStruct;
    defineMediaSources(mediaInputStruct);

    DataProcessingConditions dataProcessingConditions;
    defineProcessingConditions(frameBatchSize, featureExtractingThreshold, requiredExtractedPointsCount,
        requiredMatchedPointsCount, matcherType, dataProcessingConditions);

    Mat lastGoodFrame;
    GlobalData globalDataStruct;
    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
	std::vector<BatchElement> batch; // Необходимо создавать батч тут, чтобы его не использованный хвост переносился на следующую итерацию
    initTemporalImageDataDeque(temporalImageDataDeque);

    // Process the first pair of frames
    if (!processingFirstPairFrames(
		mediaInputStruct, dataProcessingConditions, batch,
		temporalImageDataDeque, lastGoodFrame, globalDataStruct.spatialPoints
	)) {
        // Error message if at least two good frames are not found in the video
        std::cerr << "Couldn't find at least two good frames in video" << std::endl;
        exit(-1);
    }

	std::vector<Mat> rotations;
	rotations.push_back(temporalImageDataDeque.at(0).rotation.clone());
	globalDataStruct.spatialCameraPositions.push_back(temporalImageDataDeque.at(0).motion.clone());
	rotations.push_back(temporalImageDataDeque.at(1).rotation.clone());
	globalDataStruct.spatialCameraPositions.push_back(temporalImageDataDeque.at(1).motion.clone());

    int requiredframesCount = 5;
    int framesCount = 0;
    std::vector<Mat> imagesForReconstruct;
    imagesForReconstruct.push_back(lastGoodFrame.clone());

    int lastGoodFrameIdx = 1;
    Mat nextGoodFrame;


    bool hasVideoGoodFrames;
    while (true) {
		logStreams.mainReportStream << std::endl << "================================================================" << std::endl << std::endl;

        // Find the next good frame batch
        hasVideoGoodFrames = findGoodVideoFrameFromBatch(
			mediaInputStruct, dataProcessingConditions, batch,
			lastGoodFrame, nextGoodFrame,
			temporalImageDataDeque.at(lastGoodFrameIdx).allExtractedFeatures,
			temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
			temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches
		);
        if (!hasVideoGoodFrames) {
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
        rotations.push_back(temporalImageDataDeque.at(lastGoodFrameIdx+1).rotation.clone());
		globalDataStruct.spatialCameraPositions.push_back(temporalImageDataDeque.at(lastGoodFrameIdx+1).motion.clone());


		pushNewSpatialPoints(temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches,
							 temporalImageDataDeque.at(lastGoodFrameIdx).correspondSpatialPointIdx,
							 temporalImageDataDeque.at(lastGoodFrameIdx+1).correspondSpatialPointIdx,
							 newSpatialPoints, globalDataStruct.spatialPoints);

        // Update last good frame
        lastGoodFrame = nextGoodFrame.clone();  // будет ли в будущем освобождаться от старых значений nextGoodFrame?
        imagesForReconstruct.push_back(lastGoodFrame.clone());

        if (lastGoodFrameIdx == OPTIMAL_DEQUE_SIZE - 2) {
            temporalImageDataDeque.pop_front();
			temporalImageDataDeque.push_back({});
        } else {
            lastGoodFrameIdx++;
        }
        framesCount++;

    }
    vizualizePointsAndCameras(globalDataStruct.spatialPoints,
                                rotations,
                                globalDataStruct.spatialCameraPositions,
                                dataProcessingConditions.calibrationMatrix);

	rawOutput(globalDataStruct.spatialPoints, logStreams.pointsStream);
	logStreams.pointsStream.flush();
}
