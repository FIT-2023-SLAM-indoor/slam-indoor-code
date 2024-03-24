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
void defineProcessingEnvironment(
    MediaSources &mediaInputStruct, 
    DataProcessingConditions &dataProcessingConditions
);


/**
 * Написать документацию!!!
*/
bool getNextFrame(MediaSources &mediaInputStruct, Mat &nextFrame);


/**
 * Написать документацию!!!
*/
bool findFirstGoodVideoFrameAndFeatures(
    MediaSources &mediaInputStruct,
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures
);


/**
 * Написать документацию!!!
*/
void getObjAndImgPoints(
    std::vector<DMatch> &matches,
    std::vector<int> &correspondSpatialPointIdx,
    std::vector<Point3f> &spatialPoints,
    std::vector<KeyPoint> &extractedFeatures,
    std::vector<Point3f> &objPoints,
    std::vector<Point2f> &imgPoints
);


/**
 * Написать документацию!!!
*/
void computeTransformationAndMaskPoints(
    DataProcessingConditions &dataProcessingConditions,
    Mat &chiralityMask,
    TemporalImageData &prevFrameData,
    TemporalImageData &newFrameData,
    std::vector<Point2f> &extractedPointCoords1,
    std::vector<Point2f> &extractedPointCoords2
);


/**
 * Написать документацию!!!
*/
void defineCorrespondenceIndices(
    DataProcessingConditions &dataProcessingConditions,
    Mat &chiralityMask,
    TemporalImageData &prevFrameData,
    TemporalImageData &newFrameData
);


/**
 * Написать документацию!!!
*/
void pushNewSpatialPoints(
	const std::vector<DMatch> &matches,
	std::vector<int> &prevFrameCorrespondingIndices,
	std::vector<int> &currFrameCorrespondingIndices,
	const std::vector<Point3f> &newSpatialPoints,
	std::vector<Point3f> &allSpatialPoints
); 
