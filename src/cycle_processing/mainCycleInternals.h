#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;

/**
 * Struct containing conditions for data processing.
 */
struct DataProcessingConditions {
    Mat calibrationMatrix;            // Calibration matrix for camera.
    Mat distortionCoeffs;             // Distortion coefficients for camera.
    int frameBatchSize;               // Size of batch of frames.
    int featureExtractingThreshold;   // Threshold for feature extraction.
    int requiredExtractedPointsCount; // Required number of extracted points in frame.
    int requiredMatchedPointsCount;   // Required number of matched points in frame.
    int matcherType;                  // Type of descriptor matcher to use.
};


/**
 * Функция, задающая значение предусловиям обработки медиа данных.
 *
 * @param [out] mediaInputStruct
 * @param [out] dataProcessingConditions
*/
void defineProcessingEnvironment(MediaSources &mediaInputStruct, 
                                 DataProcessingConditions &dataProcessingConditions);