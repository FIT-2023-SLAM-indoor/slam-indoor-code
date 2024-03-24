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

	GlobalData globalDataStruct;
	while(mainCycle(globalDataStruct));

	/* Что-то делаем с GlobalData */

	closeLogsStreams();
    return 0;
}