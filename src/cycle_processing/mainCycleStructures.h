#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

/**
 * This structure is designed to handle both video and photo data sources,
 * allowing the same functions to process either type of media without the need for overriding.
 */
struct MediaSources {
	VideoCapture frameSequence;
	std::vector<String> photosPaths;
	bool isPhotoProcessing;
};

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
 * Struct containing temporal image data.
 */
struct TemporalImageData {
	std::vector<KeyPoint> allExtractedFeatures;       // All extracted features of the i-th frame.
	std::vector<Vec3b> colorsForAllExtractedFeatures; // Colors for all extracted features of the i-th frame.
	std::vector<DMatch> allMatches;                   // Matches between features of frames i-1 and i.
	Mat rotation;                                     // Rotation between frames i-1 and i.
	Mat motion;                                       // Motion between frames i-1 and i.
	std::vector<int> correspondSpatialPointIdx;       // Labels field for three-dimensional points.
};

struct GlobalData {
	std::vector<Point3f> spatialPoints;
	std::vector<Vec3b> spatialPointsColors;
	std::vector<Mat> spatialCameraPositions;
	std::vector<Mat> cameraRotations;
};

/**
 * Structure for batch elements.
 */
typedef struct BatchElement {
	Mat frame;
	std::vector<KeyPoint> features;
} BatchElement;