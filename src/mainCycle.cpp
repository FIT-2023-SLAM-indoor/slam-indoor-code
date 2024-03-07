#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "cameraCalibration.h"
#include "fastExtractor.h"
#include "featureMatching.h"
#include "IOmisc.h"
#include "mainCycle.h"

using namespace cv;

const int OPTIMIZE_DEQUE_SIZE = 8;

/*
Список того, на что я пока решил забить:
1) Сохранение цветов фич - надо немного переписать Extractor (ну или добавить это в функции поиска frame-ов)
2) Выбор применения Undistortion-а к кадрам - сейчас он просто применяется
3) Обработка крайних случаев: этот код нужно хорошо отревьюить, я толком не думал про небезопасные места
4) Надо ли постоянно вызывать .clear() при создании новых объектов?
5) Указывать числам тип в виде uint32_t вместо int
*/

/*
Уже нужно рефакторить этот код. Нужна структура под данные камеры.
Аналогично можно поступить с другими данными, которые встречаются часто вместе.
*/

bool findFirstGoodVideoFrameAndFeatures(
    VideoCapture &frameSequence, 
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame, std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (frameSequence.read(candidateFrame)) {
        undistort(
            candidateFrame, goodFrame, 
            dataProcessingConditions.calibrationMatrix, 
            dataProcessingConditions.distortionCoeffs);
        fastExtractor(
            goodFrame, goodFrameFeatures, 
            dataProcessingConditions.featureExtractingThreshold);

        if (goodFrameFeatures.size() >= dataProcessingConditions.requiredExtractedPointsCount) {
            return true;
        }
    }

    return false;
}


void matchFramesPairFeatures(
	Mat& firstFrame,
	Mat& secondFrame,
	std::vector<KeyPoint>& firstFeatures,
	std::vector<KeyPoint>& secondFeatures,
	DataProcessingConditions &dataProcessingConditions,
	std::vector<DMatch>& matches
) {
	Mat firstDescriptor;
	extractDescriptor(
        firstFrame, firstFeatures, 
        dataProcessingConditions.matcherType, firstDescriptor);

	Mat secondDescriptor;
	extractDescriptor(secondFrame,secondFeatures, 
        dataProcessingConditions.matcherType, secondDescriptor);

	matchFeatures(
        firstDescriptor, secondDescriptor, matches, 
        dataProcessingConditions.matcherType, dataProcessingConditions.radius);
}


void createVideoFrameBatch(
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
    VideoCapture &frameSequence,
    DataProcessingConditions &dataProcessingConditions,
    int frameBatchSize,
    Mat &previousFrame, Mat &newGoodFrame,
    std::vector<KeyPoint> &previousFeatures, 
    std::vector<KeyPoint> &newFeatures,
    std::vector<DMatch> &matches)
{
    std::vector<Mat> frameBatch;
    createVideoFrameBatch(frameSequence, frameBatchSize, frameBatch);

    Mat candidateFrame;
    for (int frameIndex = frameBatch.size() - 1; frameIndex >= 0; frameIndex--) {
        candidateFrame = frameBatch.at(frameIndex);
        undistort(
            candidateFrame, newGoodFrame, 
            dataProcessingConditions.calibrationMatrix, 
            dataProcessingConditions.distortionCoeffs);
        fastExtractor(
            newGoodFrame, newFeatures, 
            dataProcessingConditions.featureExtractingThreshold);
        matchFramesPairFeatures(
            previousFrame, newGoodFrame, 
            previousFeatures, newFeatures,
            dataProcessingConditions.matcherType, 
            dataProcessingConditions.radius, 
            matches);
        if (matches.size() >= dataProcessingConditions.requiredMatchedPointsCount) {
            return true;
        }
    }

    return false;
}


bool processingFirstPairFrames(
    VideoCapture &frameSequence,
    DataProcessingConditions &dataProcessingConditions,
    int frameBatchSize,
    std::deque<TemporalImageData> &temporalImageDataDeque) 
{
    Mat firstFrame;
    if (!findFirstGoodVideoFrameAndFeatures(
            frameSequence, dataProcessingConditions,
            firstFrame, temporalImageDataDeque.at(0).allExtractedFeatures)
        )
    {
        return false;
    }
    temporalImageDataDeque.at(0).rotation = Mat::eye(3, 3, CV_64FC1);
    temporalImageDataDeque.at(0).motion = Mat::zeros(3, 1, CV_64FC1);

    Mat secondFrame;
    if (!findGoodVideoFrameFromBatch(
            frameSequence, dataProcessingConditions, 
            frameBatchSize,firstFrame, secondFrame,
            temporalImageDataDeque.at(0).allExtractedFeatures,
            temporalImageDataDeque.at(1).allExtractedFeatures,
            temporalImageDataDeque.at(0).allMatches)
        )
    {
        return false;
    }

    return true;
}


void defineProcessingConditions(
    int featureExtractingThreshold, 
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius,
    DataProcessingConditions &dataProcessingConditions)
{
    Mat calibMatrixTemplate(3, 3, CV_64F);
    dataProcessingConditions.calibrationMatrix = calibMatrixTemplate;
    calibration(dataProcessingConditions.calibrationMatrix, CalibrationOption::load);

    Mat distVectorTemplate(1, 5, CV_64F);
    dataProcessingConditions.distortionCoeffs = distVectorTemplate;
    loadMatrixFromXML(CALIBRATION_PATH, dataProcessingConditions.distortionCoeffs, "DC");

    dataProcessingConditions.featureExtractingThreshold = featureExtractingThreshold;
    dataProcessingConditions.requiredExtractedPointsCount = requiredExtractedPointsCount;
    dataProcessingConditions.requiredMatchedPointsCount = requiredMatchedPointsCount;
    dataProcessingConditions.matcherType = matcherType;
    dataProcessingConditions.radius = radius;
}


void videoCycle(
    VideoCapture &frameSequence,
    int frameBatchSize, 
    int featureExtractingThreshold, 
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius)
{
    DataProcessingConditions dataProcessingConditions;
    defineProcessingConditions(
        featureExtractingThreshold, 
        requiredExtractedPointsCount,
        requiredMatchedPointsCount,
        matcherType, radius, 
        dataProcessingConditions);
    

    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMIZE_DEQUE_SIZE);
    if (!processingFirstPairFrames(
            frameSequence, dataProcessingConditions,
            frameBatchSize, temporalImageDataDeque)
        )
    {
        std::cerr << "Couldn't find at least to good frames in video" << std::endl;
        exit(-1);
    }
    
    
}