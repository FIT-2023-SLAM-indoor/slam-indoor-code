#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "cameraCalibration.h"
#include "fastExtractor.h"
#include "featureMatching.h"
#include "IOmisc.h"
#include "mainCycle.h"

#define OPTIMIZE_DEQUE_SIZE 8

using namespace cv;

/*
Список того, на что я пока решил забить:
1) Сохранение цветов фич - надо немного переписать Extractor (ну или добавить это в функции поиска frame-ов)
2) Выбор применения Undistortion-а к кадрам - сейчас он просто применяется
3) Обработка крайних случаев: этот код нужно хорошо отревьюить, я толком не думал про небезопасные места
4) Надо ли постоянно вызывать .clear() при создании новых объектов?
5) Указывать числам тип в виде uint32_t вместо int
*/

bool findFirstGoodVideoFrameAndFeatures(
    VideoCapture &frameSequence,
    Mat &calibrationMatrix, Mat &distCoeffs,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures)
{
    Mat candidateFrame;
    while (frameSequence.read(candidateFrame))
    {
        undistort(candidateFrame, goodFrame, calibrationMatrix, distCoeffs);
        fastExtractor(goodFrame, goodFrameFeatures, featureExtractingThreshold);

        if (goodFrameFeatures.size() >= requiredExtractedPointsCount)
        {
            return true;
        }
    }

    return false;
}

void createVideoFrameBatch(
    VideoCapture &frameSequence,
    int frameBatchSize,
    std::vector<Mat> &frameBatch)
{
    int currentFrameBatchSize = 0;
    Mat nextFrame;
    while (currentFrameBatchSize < frameBatchSize && frameSequence.read(nextFrame))
    {
        frameBatch.push_back(nextFrame); // мб тут нужен .clone(), но вроде не нужен - я проверял
        currentFrameBatchSize++;
    }
}

bool findGoodVideoFrameFromBatch(
    VideoCapture &frameSequence,
    Mat &calibrationMatrix, Mat &distCoeffs,
    int frameBatchSize,
    int featureExtractingThreshold,
    int requiredMatchedPointsCount,
    Mat &previousFrame,
    std::vector<KeyPoint> &previousFeatures,
    std::vector<Point2f> &previousMatchedFeatures,
    Mat &newGoodFrame,
    std::vector<cv::KeyPoint> &newFeatures,
    std::vector<Point2f> &newMatchedFeatures)
{
    std::vector<Mat> frameBatch;
    createVideoFrameBatch(frameSequence, frameBatchSize, frameBatch);

    Mat candidateFrame;
    for (int frameIndex = frameBatch.size() - 1; frameIndex >= 0; frameIndex--)
    {
        candidateFrame = frameBatch.at(frameIndex);
        undistort(candidateFrame, newGoodFrame, calibrationMatrix, distCoeffs);
        fastExtractor(newGoodFrame, newFeatures, featureExtractingThreshold);
        featureMatching(
            previousFrame, newGoodFrame, previousFeatures, newFeatures,
            newMatchedFeatures, previousMatchedFeatures);
        if (newMatchedFeatures.size() >= requiredMatchedPointsCount)
        {
            return true;
        }
    }

    return false;
}

bool processingFirstPairFrames(
    VideoCapture &frameSequence,
    Mat &calibrationMatrix, Mat &distCoeffs,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount,
    int frameBatchSize,
    std::deque<TemporalImageData> &temporalImageDataDeque) 
{

    Mat firstFrame;
    if (!findFirstGoodVideoFrameAndFeatures(
            frameSequence,
            calibrationMatrix, distCoeffs,
            featureExtractingThreshold,
            requiredExtractedPointsCount,
            firstFrame,
            temporalImageDataDeque.at(0).allExtractedFeatures))
    {
        return false;
    }
    temporalImageDataDeque.at(0).rotation = Mat::eye(3, 3, CV_64FC1);
    temporalImageDataDeque.at(0).motion = Mat::zeros(3, 1, CV_64FC1);

    Mat secondFrame;
    /*
    findGoodVideoFrameFromBatch(
        frameSequence, calibrationMatrix, distCoeffs, frameBatchSize, 
        featureExtractingThreshold, 
        requiredExtractedPointsCount, 
        firstFrame, 
        temporalImageDataDeque.at(0).allExtractedFeatures,
        )
    */
}

void videoCycle(
    VideoCapture &frameSequence,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount)
{
    Mat calibrationMatrix(3, 3, CV_64F);
    Mat distCoeffs(1, 5, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);
    loadMatrixFromXML(CALIBRATION_PATH, distCoeffs, "DC");

    std::deque<TemporalImageData> temporalImageDataDeque(OPTIMIZE_DEQUE_SIZE);
    
    
}