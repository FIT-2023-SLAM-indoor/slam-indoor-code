#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "fastExtractor.h"
#include "featureMatching.h"
#include "triangulate.h"
#include "IOmisc.h"

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


void mainCycle(
    int frameBatchSize,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius)
{
    MediaSources mediaInputStruct;
    defineMediaSources(mediaInputStruct);

    DataProcessingConditions dataProcessingConditions;
    defineProcessingConditions(featureExtractingThreshold, requiredExtractedPointsCount,
        requiredMatchedPointsCount, matcherType, radius, dataProcessingConditions);

    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
    initTemporalImageDataDeque(temporalImageDataDeque);

    Mat lastGoodFrame;
    GlobalData globalDataStruct;
    // Process the first pair of frames
    if (!processingFirstPairFrames(mediaInputStruct, frameBatchSize, dataProcessingConditions,
            temporalImageDataDeque, lastGoodFrame, globalDataStruct.spatialPoints)
    ) {
        // Error message if at least two good frames are not found in the video
        std::cerr << "Couldn't find at least two good frames in video" << std::endl;
        exit(-1);
    }

    int lastGoodFrameIdx = 1;
    Mat nextGoodFrame;
    bool hasVideoGoodFrames;
    while (true) {
		logStreams.mainReportStream << std::endl << "================================================================" << std::endl << std::endl;

        // Find the next good frame batch
        hasVideoGoodFrames = findGoodVideoFrameFromBatch(mediaInputStruct, frameBatchSize,
                                dataProcessingConditions, lastGoodFrame, nextGoodFrame,
                                temporalImageDataDeque.at(lastGoodFrameIdx).allExtractedFeatures,
                                temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
                                temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches);
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


/**
 * Defines the processing conditions for data processing.
 *
 * This function initializes the data processing conditions based on the provided parameters.
 * It sets up calibration matrix, distortion coefficients, feature extracting threshold,
 * required extracted points count, required matched points count, matcher type, and matching radius.
 *
 * @param [in] featureExtractingThreshold The threshold for feature extraction.
 * @param [in] requiredExtractedPointsCount The required number of extracted points.
 * @param [in] requiredMatchedPointsCount The required number of matched points.
 * @param [in] matcherType The type of descriptor matcher to use.
 * @param [in] radius The matching radius.
 * @param [out] dataProcessingConditions Reference to the struct to store the processing conditions.
 */
void defineProcessingConditions(
    int featureExtractingThreshold, 
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius,
    DataProcessingConditions &dataProcessingConditions)
{
    defineCalibrationMatrix(dataProcessingConditions.calibrationMatrix);
    defineDistortionCoeffs(dataProcessingConditions.distortionCoeffs);

    dataProcessingConditions.featureExtractingThreshold = featureExtractingThreshold;
    dataProcessingConditions.requiredExtractedPointsCount = requiredExtractedPointsCount;
    dataProcessingConditions.requiredMatchedPointsCount = requiredMatchedPointsCount;
    dataProcessingConditions.matcherType = matcherType;
    dataProcessingConditions.radius = radius;
}


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


/**
 * Finds the first good video frame and its features.
 *
 * This function reads frames from the video sequence captured by the provided VideoCapture object.
 * It undistorts each frame using the calibration matrix and distortion coefficients specified
 * in the data processing conditions, and then extracts features from the undistorted frame.
 * If the number of extracted features meets the required count, the function returns true and
 * stores the undistorted frame and its features in the output parameters. Otherwise, it continues
 * reading frames until a suitable frame is found or the end of the video sequence is reached.
 *
 * @param [in] frameSequence The video capture object representing the sequence of frames.
 * @param [in] dataProcessingConditions The data processing conditions including calibration matrix,
 *                                 distortion coefficients, and feature extracting threshold.
 * @param [out] goodFrame Output parameter to store the undistorted good frame.
 * @param [out] goodFrameFeatures Output parameter to store the features of the good frame.
 * @return True if a good frame with sufficient features is found, false otherwise.
 */
bool findFirstGoodVideoFrameAndFeatures(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (getNextFrame(mediaInputStruct, candidateFrame)) {
        // TODO: In future we can do UNDSTORTION here
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


/**
 * Matches features between two frames.
 *
 * This function extracts descriptors from the key points of the input frames,
 * and then matches the descriptors using the specified matcher type and radius
 * from the data processing conditions.
 *
 * @param [in] firstFrame The first input frame.
 * @param [in] secondFrame The second input frame.
 * @param [in] firstFeatures The key points of the first input frame.
 * @param [in] secondFeatures The key points of the second input frame.
 * @param [in] dataProcessingConditions The data processing conditions including matcher type and matching radius.
 * @param [out] matches Output vector to store the matches between the features of the two frames.
 */
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

    // Match the descriptors using the specified matcher type and radius
    matchFeatures(firstDescriptor, secondDescriptor, matches, 
        dataProcessingConditions.matcherType, dataProcessingConditions.radius);
}

/**
 * Fills new batch up to frameBatchSize or while new frames can be obtained.
 * <br>
 * Only frames with extracted points count greater or equal to
 * dataProcessingConditions.requiredExtractedPointsCount will be added to batch
 *
 * @param [in] mediaInputStruct
 * @param [in] dataProcessingConditions
 * @param [in] frameBatchSize
 * @param [out] frameBatch
 * @param [out] batchFeatures saved features for reason of effective findGoodVideoFrameFromBatch function working
 */
static void fillVideoFrameBatch(
    MediaSources &mediaInputStruct,
	DataProcessingConditions &dataProcessingConditions,
	int frameBatchSize, std::vector<Mat> &frameBatch, std::vector<std::vector<KeyPoint>> &batchFeatures
) {
    int currentFrameBatchSize = 0;
    Mat nextFrame;
	std::vector<KeyPoint> nextFeatures;
	int skippedFrames = 0;
	logStreams.mainReportStream << "Features count in frames added to batch: ";
    while (currentFrameBatchSize < frameBatchSize && getNextFrame(mediaInputStruct, nextFrame)) {
        // TODO: In future we can do UNDSTORTION here
		fastExtractor(nextFrame, nextFeatures,
					  dataProcessingConditions.featureExtractingThreshold);
		if (nextFeatures.size() < dataProcessingConditions.requiredExtractedPointsCount) {
			skippedFrames++;
			continue;
		}
		logStreams.mainReportStream << nextFeatures.size() << " ";
        frameBatch.push_back(nextFrame.clone());
		batchFeatures.push_back(nextFeatures);
        currentFrameBatchSize++;
    }
	logStreams.mainReportStream << std::endl << "Skipped frames while constructing batch: " << skippedFrames << std::endl;
}


/**
 * Finds a good video frame from a batch of frames for feature matching.
 *
 * This function retrieves a batch of frames from the video sequence captured by the provided
 * VideoCapture object and selects the most recent frame that meets the required criteria for
 * feature matching. It undistorts each frame, extracts features, matches them with features
 * from the previous frame, and checks if the number of matches meets the specified threshold.
 * If a frame with enough matches is found, it returns true and stores the undistorted frame,
 * its features, and the matches in the output parameters. Otherwise, it continues searching
 * through the batch of frames until a suitable frame is found or the end of the batch is reached.
 *
 * @param [in] frameSequence The video capture object representing the sequence of frames.
 * @param [in] frameBatchSize The number of frames to retrieve in each batch.
 * @param [in] dataProcessingConditions The data processing conditions including calibration matrix,
 *                                 distortion coefficients, feature extracting threshold, and
 *                                 required matched points count.
 * @param [in] previousFrame The previous frame used as a reference for feature matching.
 * @param [in] previousFeatures The features extracted from the previous frame.
 * @param [out] newGoodFrame Output parameter to store the undistorted good frame.
 * @param [out] newFeatures Output parameter to store the features extracted from the new frame.
 * @param [out] matches Output vector to store the matches between the features of the previous and new frames.
 * @return True if a good frame with sufficient matches is found, false otherwise.
 */
bool findGoodVideoFrameFromBatch(
    MediaSources &mediaInputStruct, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    Mat &previousFrame, Mat &newGoodFrame,
    std::vector<KeyPoint> &previousFeatures, 
    std::vector<KeyPoint> &newFeatures,
    std::vector<DMatch> &matches)
{
    std::vector<Mat> frameBatch;
	std::vector<std::vector<KeyPoint>> batchFeatures;
    fillVideoFrameBatch(mediaInputStruct, dataProcessingConditions, frameBatchSize, frameBatch, batchFeatures);

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
            return true;
        }
    }

    // No good frame found in the batch
    return false;
}


void defineInitialCameraPosition(TemporalImageData &initialFrame) {
    initialFrame.rotation = Mat::eye(3, 3, CV_64FC1);
    initialFrame.motion = Mat::zeros(3, 1, CV_64FC1);
}


/**
 * Masks out points based on the chirality mask.
 *
 * This function filters out points from the provided vector of 2D points based on the chirality mask.
 * The chirality mask is a binary mask where non-zero values indicate valid points to keep, and zero
 * values indicate points to remove. The function ensures that the sizes of the chirality mask and the
 * points vector match. It then creates a copy of the input points vector, clears the original vector,
 * and iterates through the chirality mask. For each non-zero value in the mask, the corresponding
 * point from the copy vector is added to the original points vector.
 *
 * @param [in] chiralityMask The chirality mask specifying which points to retain (non-zero) and which to discard (zero).
 * @param [out] extractedPoints The vector of 2D points to be filtered based on the chirality mask.
 *                        Upon completion, this vector will contain only the retained points.
 */
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


/**
 * Computes the rotation matrix and translation vector between two frames and applies a mask
 * to keep only the points with positive values in the chirality mask.
 *
 * @param [in] dataProcessingConditions Data processing conditions including calibration parameters.
 * @param [in] prevFrameData Data of the previous frame containing keypoints and other information.
 * @param [in] newFrameData Data of the new frame to which the mask is applied and which will be updated with the computation results.
 * @param [out] extractedPointCoords1 Coordinates of keypoints from the previous frame.
 * @param [out] extractedPointCoords2 Coordinates of keypoints from the new frame.
 */
void computeTransformationAndMaskPoints(
    DataProcessingConditions &dataProcessingConditions, Mat &chiralityMask,
    TemporalImageData &prevFrameData, TemporalImageData &newFrameData,
    std::vector<Point2f> &extractedPointCoords1, std::vector<Point2f> &extractedPointCoords2)
{
	getMatchedPointCoords(prevFrameData.allExtractedFeatures, newFrameData.allExtractedFeatures,
						  newFrameData.allMatches, extractedPointCoords1, extractedPointCoords2);

    estimateTransformation(
        extractedPointCoords1, extractedPointCoords2, dataProcessingConditions.calibrationMatrix,
        newFrameData.rotation, newFrameData.motion, chiralityMask);

    // Apply the chirality mask to the points from the frames
    maskoutPoints(chiralityMask, extractedPointCoords1);
    maskoutPoints(chiralityMask, extractedPointCoords2);
}


/**
 * Defines correspondence indices between keypoints in two consecutive frames.
 *
 * @param [in] dataProcessingConditions Reference to the data processing conditions.
 * @param [in] chiralityMask Matrix representing the chirality mask.
 * @param [out] prevFrameData TemporalImageData object containing data of the previous frame.
 * @param [out] newFrameData TemporalImageData object containing data of the new frame.
 */
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
    MediaSources &mediaInputStruct, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    std::deque<TemporalImageData> &temporalImageDataDeque,
    Mat &secondFrame, std::vector<Point3f> &spatialPoints)
{
    Mat firstFrame;
    if (!findFirstGoodVideoFrameAndFeatures(mediaInputStruct, dataProcessingConditions,
            firstFrame, temporalImageDataDeque.at(0).allExtractedFeatures)
    ) {
        return false;
    }
    defineInitialCameraPosition(temporalImageDataDeque.at(0));

    if (!findGoodVideoFrameFromBatch(mediaInputStruct, frameBatchSize,dataProcessingConditions,
            firstFrame, secondFrame,temporalImageDataDeque.at(0).allExtractedFeatures,
            temporalImageDataDeque.at(1).allExtractedFeatures,
            temporalImageDataDeque.at(1).allMatches)
    ) {
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


/**
 * Extracts corresponding spatial points and image points based on matches and key points.
 *
 * @param [in] matches The matches between images.
 * @param [in] correspondSpatialPointIdx Corresponding indices of spatial points.
 * @param [in] spatialPoints Spatial points.
 * @param [in] extractedFeatures Extracted key points.
 * @param [out] objPoints Corresponding spatial points.
 * @param [out] imgPoints Corresponding image key points.
 */
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


/**
 * Extracts matched keypoint coordinates from two sets of keypoints and matches.
 * This function extracts the coordinates of matched keypoints from two sets of keypoints,
 * based on the provided matches between them.
 *
 * @param [in] firstExtractedFeatures The keypoints from the first image.
 * @param [in] secondExtractedFeatures The keypoints from the second image.
 * @param [in] matches The matches between the keypoints.
 * @param [out] firstMatchedPoints Output vector to store the matched keypoints' coordinates from the first image.
 * @param [out] secondMatchedPoints Output vector to store the matched keypoints' coordinates from the second image.
 */
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
