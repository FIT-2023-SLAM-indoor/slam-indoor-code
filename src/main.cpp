#include "cameraCalibration.h"
#include "photosProcessingCycle.h"
#include "videoProcessingCycle.h"
#include "IOmisc.h"
#include "fstream"
#include "nlohmann/json.hpp"
#include "config/config.h"

#include "main_config.h"

using namespace cv;
using json = nlohmann::json;

ConfigService configService;

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Please specify path to JSON-config as the second argument" << std::endl;
		return 2;
	}
	configService.setConfigFile(argv[1]);

	std::cout << configService.getValue<std::string>(ConfigFieldEnum::CALIBRATION_PATH_);
	return 0;
#ifdef CALIB
    std::vector<String> files;
    glob("../static/for_calib/samsung-hv/*.png", files, false);
    chessboardPhotosCalibration(files, 13);
    return 0;
#endif
#ifdef PHOTOS_CYCLE
    std::vector<String> photos;
    glob(PHOTOS_PATH_PATTERN, photos, false);
    sortGlobs(photos);
    char path[] = OUTPUT_DATA_DIR;
    photosProcessingCycle(photos,
                          FT_BARRIER,
                          FT_MAX_ACCEPTABLE_DIFFERENCE,
                          FRAMES_BATCH_SIZE,
                          REQUIRED_EXTRACTED_POINTS_COUNT,
                          FEATURE_EXTRACTING_THRESHOLD,
                          path);
#else
    VideoCapture cap(VIDEO_SOURCE_PATH);
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}
	char path[] = OUTPUT_DATA_DIR;
	videoProcessingCycle(cap,
                         FT_BARRIER,
                         FT_MAX_ACCEPTABLE_DIFFERENCE,
                         FRAMES_BATCH_SIZE,
                         REQUIRED_EXTRACTED_POINTS_COUNT,
                         FEATURE_EXTRACTING_THRESHOLD,
                         path);
#endif
    return 0;
}