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

    // Match the descriptors using the specified matcher type and radius
    matchFeatures(firstDescriptor, secondDescriptor, matches, 
        dataProcessingConditions.matcherType, dataProcessingConditions.radius);
}


void fillVideoFrameBatch(
    MediaSources &mediaInputStruct, int frameBatchSize, std::vector<Mat> &frameBatch)
{
    int currentFrameBatchSize = 0;
    Mat nextFrame;
    while (currentFrameBatchSize < frameBatchSize && getNextFrame(mediaInputStruct, nextFrame)) {
        frameBatch.push_back(nextFrame);
        currentFrameBatchSize++;
    }
}


bool findGoodVideoFrameFromBatch(
    MediaSources &mediaInputStruct, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    Mat &previousFrame, Mat &newGoodFrame,
    std::vector<KeyPoint> &previousFeatures, 
    std::vector<KeyPoint> &newFeatures,
    std::vector<DMatch> &matches)
{
    std::vector<Mat> frameBatch;
    fillVideoFrameBatch(mediaInputStruct, frameBatchSize, frameBatch);

    Mat candidateFrame;
    for (int frameIndex = frameBatch.size() - 1; frameIndex >= 0; frameIndex--) {
//        candidateFrame = frameBatch.at(frameIndex);
//        undistort(candidateFrame, newGoodFrame,
//                  dataProcessingConditions.calibrationMatrix,
//                  dataProcessingConditions.distortionCoeffs);
		newGoodFrame = frameBatch.at(frameIndex).clone();
		fastExtractor(newGoodFrame, newFeatures,
                      dataProcessingConditions.featureExtractingThreshold);
        
        // Match features between the previous frame and the new frame
        matchFramesPairFeatures(previousFrame, newGoodFrame, 
                                previousFeatures, newFeatures,
                                dataProcessingConditions, matches);

		logStreams.mainReportStream << "Batch index: " << frameIndex << "; prev. extracted - curr. extracted: " <<
			previousFeatures.size() << " - " << newFeatures.size() << "; matched " << matches.size() << std::endl;
        // Check if enough matches are found
        if (matches.size() >= dataProcessingConditions.requiredMatchedPointsCount) {
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
    for (int i = 0; i < matches.size(); i++) {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;

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
		prevFrameCorrespondingIndices[queryIdx] = allSpatialPoints.size() - 1;
		currFrameCorrespondingIndices[trainIdx] = allSpatialPoints.size() - 1;
	}
}

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

    Mat lastGoodFrame;
    GlobalData globalDataStruct;
    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
    initTemporalImageDataDeque(temporalImageDataDeque);

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
