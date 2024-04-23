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
	CALIBRATION_PATH,

	USE_PHOTOS_CYCLE,
	PHOTOS_PATH_PATTERN,

	VIDEO_SOURCE_PATH,
	OUTPUT_DATA_DIR,

	USE_CUDA,

	THREADS_COUNT,

	USE_UNDISTORTION,

	REQUIRED_EXTRACTED_POINTS_COUNT,
	FEATURE_EXTRACTING_THRESHOLD,

	FRAMES_BATCH_SIZE,
	SKIP_FRAMES_FROM_BATCH_HEAD,
	USE_FIRST_FIT_IN_BATCH,

	REQUIRED_MATCHED_POINTS_COUNT,

	USE_FEATURE_TRACKER,
	USE_OWN_FT,
	FT_THREADS_COUNT_,
	USE_SAD_OWN_FT,
	USE_SSD_OWN_FT,
	FT_BARRIER_,
	FT_MAX_ACCEPTABLE_DIFFERENCE_,

	FM_SIFT_FLANN,
	FM_SIFT_BF,
	FM_ORB,

	FM_KNN_DISTANCE,

	SHOW_TRACKED_POINTS,


	RP_USE_RANSAC,
	RP_RANSAC_PROB,
	RP_RANSAC_THRESHOLD,
	RP_REQUIRED_GOOD_POINTS_PERCENT,
	RP_DISTANCE_THRESHOLD,


	USE_BUNDLE_ADJUSTMENT,
	BA_MAX_FRAMES_CNT,
	BA_THREADS_CNT,

	BA_USE_TRIVIAL_LOSS,
	BA_USE_HUBER_LOSS,
	BA_HUBER_LOSS_PARAMETER,
	BA_USE_CAUCHY_LOSS,
	BA_CAUCHY_LOSS_PARAMETER,
	BA_USE_ARCTAN_LOSS,
	BA_ARCTAN_LOSS_PARAMETER,
	BA_USE_TUKEY_LOSS,
	BA_TUKEY_LOSS_PARAMETER
};

const std::map<ConfigFieldEnum, ConfigFieldPair> configFields = {
		{CALIBRATE,                       {"calibrate",                    BOOL}},
		{VISUAL_CALIBRATION,              {"visualCalibration",            BOOL}},
		{CALIBRATION_PATH,                {"calibrationPath",              STRING}},

		{USE_PHOTOS_CYCLE,                {"usePhotosCycle",               BOOL}},
		{PHOTOS_PATH_PATTERN,             {"photosPathPattern",            STRING}},
		{VIDEO_SOURCE_PATH,               {"videoSourcePath",              STRING}},
		{OUTPUT_DATA_DIR,                 {"outputDataDir",                STRING}},

		{USE_CUDA,                        {"useCUDA",                      BOOL}},

		{THREADS_COUNT,                   {"threadsCount",                 INTEGER}},

		{USE_UNDISTORTION,                {"useUndistortion",              BOOL}},

		{REQUIRED_EXTRACTED_POINTS_COUNT, {"requiredExtractedPointsCount", INTEGER}},
		{FEATURE_EXTRACTING_THRESHOLD,    {"featureExtractingThreshold",   INTEGER}},

		{FRAMES_BATCH_SIZE,               {"framesBatchSize",              INTEGER}},
		{SKIP_FRAMES_FROM_BATCH_HEAD,     {"skipFramesFromBatchHead",      INTEGER}},
		{USE_FIRST_FIT_IN_BATCH,          {"useFirstFitInBatch",           BOOL}},

		{REQUIRED_MATCHED_POINTS_COUNT,   {"requiredMatchedPointsCount",   INTEGER}},

		{USE_FEATURE_TRACKER,             {"useFeatureTracker",           BOOL}},
		{USE_OWN_FT,                      {"useOwnFeatureTracker",        BOOL}},
		{FT_THREADS_COUNT_,               {"FTThreadsCount",              INTEGER}},
		{USE_SAD_OWN_FT,                  {"useSADOwnFT",                 BOOL}},
		{USE_SSD_OWN_FT,                  {"useSSDOwnFT",                 BOOL}},
		{FT_BARRIER_,                     {"FTBarrier",                   INTEGER}},
		{FT_MAX_ACCEPTABLE_DIFFERENCE_,   {"FTMaxAcceptableDifference",   INTEGER}},

		{FM_SIFT_FLANN,                   {"useFM-SIFT-FLANN",            BOOL}},
		{FM_SIFT_BF,                      {"useFM-SIFT-BF",               BOOL}},
		{FM_ORB,                          {"useFM-ORB",                   BOOL}},

		{FM_KNN_DISTANCE,                 {"knnMatcherDistance",          FLOATING}},

		{SHOW_TRACKED_POINTS,             {"showTrackedPoints",           BOOL}},

		{RP_USE_RANSAC,                   {"RPUseRANSAC",                 BOOL}},
		{RP_RANSAC_PROB,                  {"RPRANSACProb",                FLOATING}},
		{RP_RANSAC_THRESHOLD,             {"RPRANSACThreshold",           FLOATING}},
		{RP_REQUIRED_GOOD_POINTS_PERCENT, {"RPRequiredGoodPointsPercent", FLOATING}},
		{RP_DISTANCE_THRESHOLD,           {"RPDistanceThreshold",         FLOATING}},

		{USE_BUNDLE_ADJUSTMENT,    {"useBundleAdjustment",           BOOL}},
		{BA_MAX_FRAMES_CNT,        {"BAMaxFramesCnt",                INTEGER}},
		{BA_THREADS_CNT,           {"BAThreadsCnt",                  INTEGER}},

		{BA_USE_TRIVIAL_LOSS,      {"BAUseTrivialLossFunction",      BOOL}},
		{BA_USE_HUBER_LOSS,        {"BAUseHuberLossFunction",        BOOL}},
		{BA_HUBER_LOSS_PARAMETER,  {"BAHuberLossFunctionParameter",  FLOATING}},
		{BA_USE_CAUCHY_LOSS,       {"BAUseCauchyLossFunction",       BOOL}},
		{BA_CAUCHY_LOSS_PARAMETER, {"BACauchyLossFunctionParameter", FLOATING}},
		{BA_USE_ARCTAN_LOSS,       {"BAUseArctanLossFunction",       BOOL}},
		{BA_ARCTAN_LOSS_PARAMETER, {"BAArctanLossFunctionParameter", FLOATING}},
		{BA_USE_TUKEY_LOSS,        {"BAUseTukeyLossFunction",        BOOL}},
		{BA_TUKEY_LOSS_PARAMETER,  {"BATukeyLossFunctionParameter",  FLOATING}}
};
