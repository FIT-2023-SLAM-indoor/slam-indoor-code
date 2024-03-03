#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "mainCycle.h"

void findFirstGoodVideoFrame(
    VideoCapture &frameSequence, 
    Mat calibrationMatrix, Mat distCoeffs,
    cv::Ptr<cv::DescriptorExtractor> extractor,
    int requiredExtractedPointsCount) 
{
    Mat originalFrame;
    Mat compensatedFrame;
    Mat featureDescriptors;
    std::vector<KeyPoint> extractedFrameFeatures;

    while (frameSequence.read(originalFrame)) {
        undistort(originalFrame, compensatedFrame, calibrationMatrix, distCoeffs);
        extractor->compute(compensatedFrame, extractedFrameFeatures, featureDescriptors);
        if (extractedFrameFeatures.size() >= requiredExtractedPointsCount) {
            // тут мы сохраняем важные данные об этом первом подходящем кадре
            // пока не знаю, какие это данные: фичи или сам по себе кадр
        }
    }
}
