#include "cameraCalibration.h"
#include "IOmisc.h"
#include "fstream"

#include "mainCycle.h"
#include "config/config.h"
#include "featureMatching.h"


using namespace cv;

ConfigService configService;
LogFilesStreams logStreams;

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Please specify path to JSON-config as the second argument" << std::endl;
		return 2;
	}
	configService.setConfigFile(argv[1]);
	openLogsStreams();

	if (configService.getValue<bool>(ConfigFieldEnum::CALIBRATE)) {
		std::vector<String> files;
		glob(
			configService.getValue<std::string>(ConfigFieldEnum::PHOTOS_PATH_PATTERN_),
			files, false
		);
		chessboardPhotosCalibration(files, 13);
		return 0;
	}
	std::string path = configService.getValue<std::string>(ConfigFieldEnum::OUTPUT_DATA_DIR_);
	if (configService.getValue<bool>(ConfigFieldEnum::USE_PHOTOS_CYCLE)) {
		std::vector<String> photos;
		glob(configService.getValue<std::string>(ConfigFieldEnum::PHOTOS_PATH_PATTERN_), photos, false);
		sortGlobs(photos);
//		photosProcessingCycle(photos,
//							  FT_BARRIER,
//							  FT_MAX_ACCEPTABLE_DIFFERENCE,
//							  FRAMES_BATCH_SIZE,
//							  REQUIRED_EXTRACTED_POINTS_COUNT,
//							  FEATURE_EXTRACTING_THRESHOLD,
//							  path);
	}
	else {
		mainCycle(
			configService.getValue<int>(ConfigFieldEnum::FRAMES_BATCH_SIZE_),
			configService.getValue<int>(ConfigFieldEnum::FEATURE_EXTRACTING_THRESHOLD_),
			configService.getValue<int>(ConfigFieldEnum::REQUIRED_EXTRACTED_POINTS_COUNT_),
			configService.getValue<int>(ConfigFieldEnum::REQUIRED_EXTRACTED_POINTS_COUNT_),
			getMatcherTypeIndex(),
			configService.getValue<float>(ConfigFieldEnum::FM_SEARCH_RADIUS_)
	   );
	}
	closeLogsStreams();
    return 0;
}