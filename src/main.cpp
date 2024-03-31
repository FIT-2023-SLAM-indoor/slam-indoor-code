#include "fstream"
#include "ceres/ceres.h"

#include "config/config.h"
#include "IOmisc.h"

#include "cameraCalibration.h"

#include "cycle_processing/mainCycle.h"
#include "cycle_processing/mainCycleInternals.h"
#include "vizualizationModule.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;

ConfigService configService;
LogFilesStreams logStreams;


int main(int argc, char** argv) {
	
	if (argc < 2) {
		std::cerr << "Please specify path to JSON-config as the second argument" << std::endl;
		return 2;
	}

	configService.setConfigFile(argv[1]);
	google::InitGoogleLogging("BA");
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
	MediaSources mediaInputStruct;
	DataProcessingConditions dataProcessingConditions;
	defineProcessingEnvironment(mediaInputStruct, dataProcessingConditions);
	std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
	defineInitialCameraPosition(temporalImageDataDeque.at(0));
	do {
		/* Что-то делаем с TemporalData. А именно передаём данные о начальной позиции камеры */
	} while (mainCycle(mediaInputStruct, dataProcessingConditions, temporalImageDataDeque, globalDataStruct));

	closeLogsStreams();

	std::vector<Point3f> convertedSpatialPoints;
	for (auto point : globalDataStruct.spatialPoints)
		convertedSpatialPoints.push_back(Point3f(point));

	vizualizePointsAndCameras(convertedSpatialPoints,
							  globalDataStruct.cameraRotations,
							  globalDataStruct.spatialCameraPositions,
							  globalDataStruct.spatialPointsColors,
							  dataProcessingConditions.calibrationMatrix);

    return 0;
}