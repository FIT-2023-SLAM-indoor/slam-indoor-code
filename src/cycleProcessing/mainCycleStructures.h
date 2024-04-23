#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;

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
	Mat distortionCoeffs;             // Distortion coefficients for camera.
	bool useCUDA;
	int threadsCount;
	int frameBatchSize;               // Size of batch of frames.
	int skipFramesFromBatchHead;      // Count of frames in batch's head which won't be checked
	bool useFirstFitInBatch;
	int featureExtractingThreshold;   // Threshold for feature extraction.
	int requiredExtractedPointsCount; // Required number of extracted points in frame.
	int requiredMatchedPointsCount;   // Required number of matched points in frame.
	int matcherType;                  // Type of descriptor matcher to use.
	bool useBundleAdjustment;
	int maxProcessedFramesVectorSz;   // This parameter also specifes max frames datas cnt which will be used for bundle adjustment
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

typedef std::vector<Point3d> SpatialPointsVector;

struct GlobalData {
	SpatialPointsVector spatialPoints;
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