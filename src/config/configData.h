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
	ONLY_VIZ,

	CALIBRATE,
	VISUAL_CALIBRATION,
	CALIBRATION_PATH,

	USE_PHOTOS_CYCLE,
	PHOTOS_PATH_PATTERN,

	VIDEO_SOURCE_PATH,
	OUTPUT_DATA_DIR,

	THREADS_COUNT,

	USE_UNDISTORTION,

	REQUIRED_EXTRACTED_POINTS_COUNT,
	FEATURE_EXTRACTING_THRESHOLD,

	FRAMES_BATCH_SIZE,
	SKIP_FRAMES_FROM_BATCH_HEAD,
	USE_FIRST_FIT_IN_BATCH,

	REQUIRED_MATCHED_POINTS_COUNT,

	FM_SIFT_FLANN,
	FM_SIFT_BF,
	FM_ORB,

	FM_KNN_DISTANCE,

	RP_USE_RANSAC,
	RP_RANSAC_PROB,
	RP_RANSAC_THRESHOLD,
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
		{ONLY_VIZ,                        {"onlyViz",                      BOOL}},

		{CALIBRATE,                       {"calibrate",                    BOOL}},
		{VISUAL_CALIBRATION,              {"visualCalibration",            BOOL}},
		{CALIBRATION_PATH,                {"calibrationPath",              STRING}},

		{USE_PHOTOS_CYCLE,                {"usePhotosCycle",               BOOL}},
		{PHOTOS_PATH_PATTERN,             {"photosPathPattern",            STRING}},
		{VIDEO_SOURCE_PATH,               {"videoSourcePath",              STRING}},
		{OUTPUT_DATA_DIR,                 {"outputDataDir",                STRING}},

		{THREADS_COUNT,                   {"threadsCount",                 INTEGER}},

		{USE_UNDISTORTION,                {"useUndistortion",              BOOL}},

		{REQUIRED_EXTRACTED_POINTS_COUNT, {"requiredExtractedPointsCount", INTEGER}},
		{FEATURE_EXTRACTING_THRESHOLD,    {"featureExtractingThreshold",   INTEGER}},

		{FRAMES_BATCH_SIZE,               {"framesBatchSize",              INTEGER}},
		{SKIP_FRAMES_FROM_BATCH_HEAD,     {"skipFramesFromBatchHead",      INTEGER}},
		{USE_FIRST_FIT_IN_BATCH,          {"useFirstFitInBatch",           BOOL}},

		{REQUIRED_MATCHED_POINTS_COUNT,   {"requiredMatchedPointsCount",   INTEGER}},

		{FM_SIFT_FLANN,                   {"useFM-SIFT-FLANN",            BOOL}},
		{FM_SIFT_BF,                      {"useFM-SIFT-BF",               BOOL}},
		{FM_ORB,                          {"useFM-ORB",                   BOOL}},

		{FM_KNN_DISTANCE,                 {"knnMatcherDistance",          FLOATING}},

		{RP_USE_RANSAC,                   {"RPUseRANSAC",                 BOOL}},
		{RP_RANSAC_PROB,                  {"RPRANSACProb",                FLOATING}},
		{RP_RANSAC_THRESHOLD,             {"RPRANSACThreshold",           FLOATING}},
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
