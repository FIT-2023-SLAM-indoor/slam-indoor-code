#include "fstream"
#include "ceres/ceres.h"

#include "config/config.h"
#include "IOmisc.h"
#include "cameraCalibration.h"

#include "cycleProcessing/mainCycle.h"
#include "cycleProcessing/mainCycleInternals.h"
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
	std::string path = configService.getValue<std::string>(ConfigFieldEnum::OUTPUT_DATA_DIR);

	MediaSources mediaInputStruct;
	DataProcessingConditions dataProcessingConditions;
	defineProcessingEnvironment(mediaInputStruct, dataProcessingConditions);
	Mat calibrationMatrix;
	defineCalibrationMatrix(calibrationMatrix);
	GlobalData globalDataStruct;
	std::deque<TemporalImageData> oldTempImageDataDeque;
	int lastFrameOfLaunchId = -1;
	do {
		std::deque<TemporalImageData> newTempImageDataDeque(OPTIMAL_DEQUE_SIZE);
		defineInitialCameraPosition(newTempImageDataDeque.at(0));
		/*
		std::deque<TemporalImageData> oldTempImageData = newTempImageDataDeque;
		newTempImageDataDeque.clear();
		newTempImageDataDeque.resize(OPTIMAL_DEQUE_SIZE);
		newTempImageDataDeque.at(0).rotation = oldTempImageData.at(lastFrameOfLaunchId).rotation;
		newTempImageDataDeque.at(0).motion = oldTempImageData.at(lastFrameOfLaunchId).motion;
		*/

		GlobalData newGlobalData;
		lastFrameOfLaunchId = mainCycle(mediaInputStruct, calibrationMatrix,
										dataProcessingConditions, newTempImageDataDeque,
										newGlobalData);
		oldTempImageDataDeque = newTempImageDataDeque;
		insertNewGlobalData(globalDataStruct, newGlobalData);
	} while (lastFrameOfLaunchId > 0);

	rawOutput(globalDataStruct.spatialPoints, logStreams.pointsStream);
	logStreams.pointsStream.flush();

	// Kostil for Points3f in visualizer
	std::vector<Point3f> convertedSpatialPoints;
	for (auto point : globalDataStruct.spatialPoints)
		convertedSpatialPoints.push_back(Point3f(point));

	vizualizePointsAndCameras(convertedSpatialPoints,
							  globalDataStruct.cameraRotations,
							  globalDataStruct.spatialCameraPositions,
							  globalDataStruct.spatialPointsColors,
							  calibrationMatrix);

	closeLogsStreams();

    return 0;
}
