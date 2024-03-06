#pragma once
#include "fstream"

#include <opencv2/core.hpp>

using namespace cv;

/**
/**
 * Struct containing conditions for data processing.
 */
typedef struct DataProcessingConditions {
    Mat calibrationMatrix;             // Calibration matrix for camera.
    Mat distanceCoeffs;                // Distortion coefficients for camera.
    int featureExtractingThreshold;    // Threshold for feature extraction.
    int requiredExtractedPointsCount;  // Required number of extracted points.
    int matcherType;                   // Type of descriptor matcher to use.
    float radius;                      // Matching radius.
} DataProcessingConditions;

/**
 *
 */
typedef struct LogFilesStreams {
    std::ofstream mainReportStream;
    std::ofstream pointsStream;
    std::ofstream poseStream;
    std::ofstream poseHandyStream;
    std::ofstream poseGlobalMltStream;
} LogFilesStreams;

typedef struct TemporalImageData {
	std::vector<KeyPoint> allExtractedFeatures; // Все заэкстратенные фичи i-го кадра
	std::vector<Vec3b> colorsForAllExtractedFeatures; // цвета j-й фичи i-го кадра
	std::vector<DMatch> allMatches; // Матчи между фичами кадров i-1 и i
	Mat rotation; // Вращения между кадрами i-1 и i
	Mat motion; // Сдвиги между кадрами i-1 и i
	std::vector<int> correspondSpatialPointIdx; // Поле меток для трёхмерных точек (см. ниже)
} TemporalImageData;

typedef struct GlobalData {
	std::vector<Point3f> spatialPoints;
	std::vector<Vec3b> spatialPointsColors;
	std::vector<Mat> spatialCameraPositions;
} GlobalData;
