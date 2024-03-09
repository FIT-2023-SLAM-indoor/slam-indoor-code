#pragma once
#include "fstream"

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

typedef struct LogFilesStreams {
    std::ofstream mainReportStream;
    std::ofstream pointsStream;
    std::ofstream poseStream;
    std::ofstream poseHandyStream;
    std::ofstream poseGlobalMltStream;
} LogFilesStreams;

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
 * Defines the processing conditions for data processing.
 *
 * This function initializes the data processing conditions based on the provided parameters.
 * It sets up calibration matrix, distortion coefficients, feature extracting threshold,
 * required extracted points count, required matched points count, matcher type, and matching radius.
 *
 * @param [in] featureExtractingThreshold The threshold for feature extraction.
 * @param [in] requiredExtractedPointsCount The required number of extracted points.
 * @param [in] requiredMatchedPointsCount The required number of matched points.
 * @param [in] matcherType The type of descriptor matcher to use.
 * @param [in] radius The matching radius.
 * @param [out] dataProcessingConditions Reference to the struct to store the processing conditions.
 */
void defineProcessingConditions(
    int featureExtractingThreshold, 
    int requiredExtractedPointsCount,
    int requiredMatchedPointsCount,
    int matcherType, float radius,
    DataProcessingConditions &dataProcessingConditions
);


/**
 * Finds the first good video frame and its features.
 *
 * This function reads frames from the video sequence captured by the provided VideoCapture object.
 * It undistorts each frame using the calibration matrix and distortion coefficients specified
 * in the data processing conditions, and then extracts features from the undistorted frame.
 * If the number of extracted features meets the required count, the function returns true and
 * stores the undistorted frame and its features in the output parameters. Otherwise, it continues
 * reading frames until a suitable frame is found or the end of the video sequence is reached.
 *
 * @param [in] frameSequence The video capture object representing the sequence of frames.
 * @param [in] dataProcessingConditions The data processing conditions including calibration matrix,
 *                                 distortion coefficients, and feature extracting threshold.
 * @param [out] goodFrame Output parameter to store the undistorted good frame.
 * @param [out] goodFrameFeatures Output parameter to store the features of the good frame.
 * @return True if a good frame with sufficient features is found, false otherwise.
 */
bool findFirstGoodVideoFrameAndFeatures(
    VideoCapture &frameSequence,
    DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
    std::vector<KeyPoint> &goodFrameFeatures
);


/**
 * Matches features between two frames.
 *
 * This function extracts descriptors from the key points of the input frames,
 * and then matches the descriptors using the specified matcher type and radius
 * from the data processing conditions.
 *
 * @param [in] firstFrame The first input frame.
 * @param [in] secondFrame The second input frame.
 * @param [in] firstFeatures The key points of the first input frame.
 * @param [in] secondFeatures The key points of the second input frame.
 * @param [in] dataProcessingConditions The data processing conditions including matcher type and matching radius.
 * @param [out] matches Output vector to store the matches between the features of the two frames.
 */
void matchFramesPairFeatures(
    Mat& firstFrame,
    Mat& secondFrame,
    std::vector<KeyPoint>& firstFeatures,
    std::vector<KeyPoint>& secondFeatures,
    DataProcessingConditions &dataProcessingConditions,
    std::vector<DMatch>& matches
); 