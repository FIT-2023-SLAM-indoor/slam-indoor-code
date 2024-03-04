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
	CALIBRATION_PATH_,
	EXTRACTED_FEATURES_THRESHOLD,
	MATCHING_RADIUS
};

const std::map<ConfigFieldEnum, ConfigFieldPair> configFields = {
		{CALIBRATE,{"calibrate", BOOL}},
		{CALIBRATION_PATH_,{"calibrationPath", STRING}},
		{EXTRACTED_FEATURES_THRESHOLD,{"extractedFeaturesThreshold", INTEGER}},
		{MATCHING_RADIUS,{"matchingRadius", FLOATING}}
};
