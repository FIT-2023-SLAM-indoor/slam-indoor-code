#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "mainCycle.h"
#include "fastExtractor.h"


/*
Список того, на что я решил забить:
1) Сохранение цветов фич
2) Выбор применения Undistortion-а к кадрам - сейчас он просто применяется

*/


void findFirstGoodVideoFrameAndFeatures(
    VideoCapture &frameSequence, 
    Mat &calibrationMatrix, Mat &distCoeffs,
    int featureExtractingThreshold,
    int requiredExtractedPointsCount,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures) 
{
    Mat candidateFrame;
    while (frameSequence.read(candidateFrame)) {
        undistort(candidateFrame, goodFrame, calibrationMatrix, distCoeffs);
        fastExtractor(goodFrame, goodFrameFeatures, featureExtractingThreshold);
        
        if (goodFrameFeatures.size() >= requiredExtractedPointsCount) {
            break;
        }
    }
}


void createVideoFrameBatch(
    VideoCapture &frameSequence,
    int frameBatchSize,
    std::vector<Mat> &frameBatch)
{
    int currentFrameBatchSize = 0;
    Mat nextFrame;
    while (currentFrameBatchSize < frameBatchSize && frameSequence.read(nextFrame)) {
        frameBatch.push_back(nextFrame.clone());
        currentFrameBatchSize++;
    }
}


void findGoodVideoFrameFromBatch(
    )
{
    
}