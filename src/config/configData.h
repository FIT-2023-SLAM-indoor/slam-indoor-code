#pragma once
#include "map"

enum ConfigFieldType {
	BOOL,
	STRING,
	INTEGER,
	FLOATING
};

typedef struct ConfigFieldPair {
	std::string key;
	ConfigFieldType type;
} ConfigFieldPair;

enum ConfigFieldEnum {
	CALIBRATE,
	VISUAL_CALIBRATION,
	CALIBRATION_PATH_,

	USE_PHOTOS_CYCLE,
	PHOTOS_PATH_PATTERN_,

	VIDEO_SOURCE_PATH_,
	OUTPUT_DATA_DIR_,

	USE_UNDISTORTION_,

	REQUIRED_EXTRACTED_POINTS_COUNT_,
	FEATURE_EXTRACTING_THRESHOLD_,

	FRAMES_BATCH_SIZE_,

	REQUIRED_MATCHED_POINTS_COUNT,

	USE_FEATURE_TRACKER,
	USE_OWN_FT,
	FT_THREADS_COUNT_,
	USE_SAD_OWN_FT,
	USE_SSD_OWN_FT,
	FT_BARRIER_,
	FT_MAX_ACCEPTABLE_DIFFERENCE_,

	FM_SIFT_FLANN_,
	FM_SIFT_BF_,
	FM_ORB_,
	FM_SEARCH_RADIUS_,

	SHOW_TRACKED_POINTS,


	RP_USE_RANSAC,
	RP_RANSAC_PROB,
	RP_RANSAC_THRESHOLD,
	RP_REQUIRED_GOOD_POINTS_PERCENT,
	RP_DISTANCE_THRESHOLD
};

const std::map<ConfigFieldEnum, ConfigFieldPair> configFields = {
		{CALIBRATE,{"calibrate", BOOL}},
		{VISUAL_CALIBRATION,{"visualCalibration", BOOL}},
		{CALIBRATION_PATH_,{"calibrationPath", STRING}},

		{USE_PHOTOS_CYCLE,{"usePhotosCycle", BOOL}},
		{PHOTOS_PATH_PATTERN_,{"photosPathPattern", STRING}},
		{VIDEO_SOURCE_PATH_,{"videoSourcePath", STRING}},
		{OUTPUT_DATA_DIR_,{"outputDataDir", STRING}},

		{USE_UNDISTORTION_,{"useUndistortion", BOOL}},

		{REQUIRED_EXTRACTED_POINTS_COUNT_,{"requiredExtractedPointsCount", INTEGER}},
		{FEATURE_EXTRACTING_THRESHOLD_,{"featureExtractingThreshold", INTEGER}},

		{FRAMES_BATCH_SIZE_,{"framesBatchSize", INTEGER}},

		{REQUIRED_MATCHED_POINTS_COUNT,{"requiredMatchedPointsCount", INTEGER}},

		{USE_FEATURE_TRACKER,{"useFeatureTracker", BOOL}},
		{USE_OWN_FT,{"useOwnFeatureTracker", BOOL}},
		{FT_THREADS_COUNT_,{"FTThreadsCount", INTEGER}},
		{USE_SAD_OWN_FT, {"useSADOwnFT", BOOL}},
		{USE_SSD_OWN_FT, {"useSSDOwnFT", BOOL}},
		{FT_BARRIER_,{"FTBarrier", INTEGER}},
		{FT_MAX_ACCEPTABLE_DIFFERENCE_,{"FTMaxAcceptableDifference", INTEGER}},

		{FM_SIFT_FLANN_,{"useFM-SIFT-FLANN", BOOL}},
		{FM_SIFT_BF_,{"useFM-SIFT-BF", BOOL}},
		{FM_ORB_,{"useFM-ORB", BOOL}},

		{FM_SEARCH_RADIUS_,{"featureMatchingRadius", FLOATING}},

		{SHOW_TRACKED_POINTS,{"showTrackedPoints", BOOL}},

		{RP_USE_RANSAC,{"RPUseRANSAC", BOOL}},
		{RP_RANSAC_PROB,{"RPRANSACProb", FLOATING}},
		{RP_RANSAC_THRESHOLD,{"RPRANSACThreshold", FLOATING}},
		{RP_REQUIRED_GOOD_POINTS_PERCENT,{"RPRequiredGoodPointsPercent", FLOATING}},
		{RP_DISTANCE_THRESHOLD,{"RPDistanceThreshold", FLOATING}},
};
