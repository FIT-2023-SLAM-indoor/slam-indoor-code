#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "fastExtractor.h"
#include "featureMatching.h"
#include "triangulate.h"
#include "IOmisc.h"
#include "mainCycle.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;

/*
Список того, на что я пока решил забить:
1) Сохранение цветов фич - надо немного переписать Extractor (ну или добавить это в функции поиска frame-ов)
2) Режим обработки последовательности фотографий
3) Выбор применения Undistortion-а к кадрам - сейчас он просто применяется
4) Обработка крайних случаев: этот код нужно хорошо отревьюить, я толком не думал про небезопасные места
*/


void defineCalibrationMatrix(Mat &calibrationMatrix) {
    // Сreating a new matrix or changing the type and size of an existing one
    calibrationMatrix.create(3, 3, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);
}


void defineDistortionCoeffs(Mat &distortionCoeffs) {
    // Сreating a new matrix or changing the type and size of an existing one
    distortionCoeffs.create(1, 5, CV_64F);
    loadMatrixFromXML(CALIBRATION_PATH, distortionCoeffs, "DC");
}


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


bool findFirstGoodVideoFrameAndFeatures(
    VideoCapture &frameSequence,
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (frameSequence.read(candidateFrame)) {
        undistort(candidateFrame, goodFrame, 
            dataProcessingConditions.calibrationMatrix,
            dataProcessingConditions.distortionCoeffs);
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
    VideoCapture &frameSequence, int frameBatchSize, std::vector<Mat> &frameBatch)
{
    int currentFrameBatchSize = 0;
    Mat nextFrame;
    while (currentFrameBatchSize < frameBatchSize && frameSequence.read(nextFrame)) {
        frameBatch.push_back(nextFrame);
        currentFrameBatchSize++;
    }
}


bool findGoodVideoFrameFromBatch(
    VideoCapture &frameSequence, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    Mat &previousFrame, Mat &newGoodFrame,
    std::vector<KeyPoint> &previousFeatures, 
    std::vector<KeyPoint> &newFeatures,
    std::vector<DMatch> &matches)
{
    std::vector<Mat> frameBatch;
    fillVideoFrameBatch(frameSequence, frameBatchSize, frameBatch);

    Mat candidateFrame;
    for (int frameIndex = frameBatch.size() - 1; frameIndex >= 0; frameIndex--) {
        candidateFrame = frameBatch.at(frameIndex);
        undistort(candidateFrame, newGoodFrame, 
                  dataProcessingConditions.calibrationMatrix,
                  dataProcessingConditions.distortionCoeffs);
        fastExtractor(newGoodFrame, newFeatures, 
                      dataProcessingConditions.featureExtractingThreshold);
        
        // Match features between the previous frame and the new frame
        matchFramesPairFeatures(previousFrame, newGoodFrame, 
                                previousFeatures, newFeatures,
                                dataProcessingConditions, matches);
        
        // Check if enough matches are found
        if (matches.size() >= dataProcessingConditions.requiredMatchedPointsCount) {
            return true;
        }
    }

    // No good frame found in the batch
    return false;
}


void defineInitialCameraPosition(TemporalImageData initialFrame) {
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
    // Convert keypoints from the previous frame into coordinates
    KeyPoint::convert(prevFrameData.allExtractedFeatures, extractedPointCoords1);
    KeyPoint::convert(newFrameData.allExtractedFeatures, extractedPointCoords2);

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
    VideoCapture &frameSequence, int frameBatchSize,
    DataProcessingConditions &dataProcessingConditions,
    std::deque<TemporalImageData> &temporalImageDataDeque,
    Mat &secondFrame, std::vector<Point3f> &spatialPoints)
{
    Mat firstFrame;
    if (!findFirstGoodVideoFrameAndFeatures(frameSequence, dataProcessingConditions,
            firstFrame, temporalImageDataDeque.at(0).allExtractedFeatures)
        )
    {
        return false;
    }
    defineInitialCameraPosition(temporalImageDataDeque.at(0));

    if (!findGoodVideoFrameFromBatch(frameSequence, frameBatchSize,dataProcessingConditions,
            firstFrame, secondFrame,temporalImageDataDeque.at(0).allExtractedFeatures,
            temporalImageDataDeque.at(1).allExtractedFeatures,
            temporalImageDataDeque.at(1).allMatches)
        )
    {
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

// Получаем те из имеющихся трехмерных точек, которые были заматчены на новом кадре 
void getObjAndImgPoints(
    std::vector<DMatch> &matches,
    std::vector<int> &correspondSpatialPointIdx, 
    std::vector<Point3f> &spatialPoints, 
    std::vector<KeyPoint> &extractedFeatures,
    std::vector<Point3f> &objPoints,
    std::vector<Point2f> &imgPoints)
{
    objPoints.clear();
    imgPoints.clear();
 
    for (int i = 0; i < matches.size(); ++i)
    {
        int query_idx = matches[i].queryIdx;
        int train_idx = matches[i].trainIdx;
 
        int struct_idx = correspondSpatialPointIdx[query_idx];
        if (struct_idx >= 0) {
            objPoints.push_back(spatialPoints[struct_idx]);
            imgPoints.push_back(extractedFeatures[train_idx].pt);
        }
    }
}


// У функции и её параметров плохие имена/названия
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


/* НАДО ЧИСТИТЬ ВСЕ СТРУКТУРЫ СРАЗУ ПОСЛЕ ИХ ОБЪЯВЛЕНИЯ */
void videoCycle(
    VideoCapture &frameSequence,
    int frameBatchSize, 
    int featureExtractingThreshold, 
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius)
{
    DataProcessingConditions dataProcessingConditions;
    defineProcessingConditions(featureExtractingThreshold, requiredExtractedPointsCount,
        requiredMatchedPointsCount, matcherType, radius, dataProcessingConditions);

    Mat lastGoodFrame;
    GlobalData globalDataStruct;
    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
    if (!processingFirstPairFrames(frameSequence, frameBatchSize,dataProcessingConditions,
            temporalImageDataDeque,lastGoodFrame,globalDataStruct.spatialPoints)
        )
    {
        std::cerr << "Couldn't find at least to good frames in video" << std::endl;
        exit(-1);
    }

    bool hasVideoGoodFrames;
    int lastGoodFrameIdx = 1;
    Mat nextGoodFrame;
    while (true) {
        bool hasVideoGoodFrames = findGoodVideoFrameFromBatch(frameSequence, frameBatchSize,
                            dataProcessingConditions,lastGoodFrame, nextGoodFrame,
                            temporalImageDataDeque.at(lastGoodFrameIdx).allExtractedFeatures,
                            temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
                            temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches);
        if (!hasVideoGoodFrames) {
            std::cerr << "No good frames in batch. Stop video processing" << std::endl;
            break;  // Может не брейк, а что-то другое
        }

        std::vector<Point3f> objPoints;
        std::vector<Point2f> imgPoints;
        getObjAndImgPoints(
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches,
            temporalImageDataDeque.at(lastGoodFrameIdx).correspondSpatialPointIdx,
            globalDataStruct.spatialPoints,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
            objPoints, imgPoints);

        // Решаем матрицу преобразования
        Mat rotationVector;
        solvePnPRansac(
            objPoints, imgPoints, 
            dataProcessingConditions.calibrationMatrix, 
            noArray(), rotationVector, 
            temporalImageDataDeque.at(lastGoodFrameIdx+1).motion);
        // Преобразуем вектор вращения в матрицу вращения
        Rodrigues(rotationVector, temporalImageDataDeque.at(lastGoodFrameIdx+1).rotation);
        // 3D-реконструкция на основе полученных ранее значений R и T
        std::vector<Point2f> matchedPointCoords1;
        std::vector<Point2f> matchedPointCoords2;
        std::vector<Point3f> newSpatialPoints;
        getMatchedPointCoords(temporalImageDataDeque.at(lastGoodFrameIdx).allExtractedFeatures,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allExtractedFeatures,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).allMatches,
            matchedPointCoords1, matchedPointCoords2);
        reconstruct(dataProcessingConditions.calibrationMatrix,
            temporalImageDataDeque.at(lastGoodFrameIdx).rotation,
            temporalImageDataDeque.at(lastGoodFrameIdx).motion,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).rotation,
            temporalImageDataDeque.at(lastGoodFrameIdx+1).motion, 
            matchedPointCoords1, matchedPointCoords2, newSpatialPoints);


        if (lastGoodFrameIdx == OPTIMAL_DEQUE_SIZE - 2) {
            temporalImageDataDeque.pop_front();
        } else {
            lastGoodFrameIdx++;
        }
    }
}