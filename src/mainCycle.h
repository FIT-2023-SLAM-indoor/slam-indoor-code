#pragma once
#include "fstream"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;

/**
 * Struct containing conditions for data processing.
 */
typedef struct DataProcessingConditions {
    Mat calibrationMatrix;            // Calibration matrix for camera.
    Mat distortionCoeffs;             // Distortion coefficients for camera.
    int featureExtractingThreshold;   // Threshold for feature extraction.
    int requiredExtractedPointsCount; // Required number of extracted points in frame.
    int requiredMatchedPointsCount;   // Required number of of matched points in frame.
    int matcherType;                  // Type of descriptor matcher to use.
    float radius;                     // Matching radius.
} DataProcessingConditions;

/**
 * This structure is designed to handle both video and photo data sources,
 * allowing the same functions to process either type of media without the need for overriding.
*/
typedef struct MediaSources {
    VideoCapture frameSequence;
    std::vector<String> photosPaths;
    bool isPhotoProcessing;
} MediaSources;

/**
 * Struct containing temporal image data.
 */
typedef struct TemporalImageData {
    std::vector<KeyPoint> allExtractedFeatures;       // All extracted features of the i-th frame.
    std::vector<Vec3b> colorsForAllExtractedFeatures; // Colors for all extracted features of the i-th frame.
    std::vector<DMatch> allMatches;                   // Matches between features of frames i-1 and i.
    Mat rotation;                                     // Rotation between frames i-1 and i.
    Mat motion;                                       // Motion between frames i-1 and i.
    std::vector<int> correspondSpatialPointIdx;       // Labels field for three-dimensional points.
} TemporalImageData;

typedef struct GlobalData {
	std::vector<Point3f> spatialPoints;
	std::vector<Vec3b> spatialPointsColors;
	std::vector<Mat> spatialCameraPositions;
} GlobalData;


/**
 * Processes a sequence of frames or photos, extracting features,
 * and performing motion analysis. This function processes a sequence of frames
 * from the provided media source. It extracts features, matches points,
 * estimates transformation matrices, and performs motion analysis to track objects
 * or patterns throughout the media image sequence.
 * 
 * @param [in] frameBatchSize The size of the frame batch to process.
 * @param [in] featureExtractingThreshold Threshold for feature extraction.
 * @param [in] requiredExtractedPointsCount Number of required extracted points.
 * @param [in] requiredMatchedPointsCount Number of required matched points.
 * @param [in] matcherType Type of feature matcher.
 * @param [in] radius Radius parameter for feature matching.
 */
void mainCycle(
    int frameBatchSize, 
    int featureExtractingThreshold, 
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius
);
