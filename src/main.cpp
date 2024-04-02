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
	GlobalData newGlobalData;
	std::deque<TemporalImageData> temporalImageDataDeque(OPTIMAL_DEQUE_SIZE);
	defineInitialCameraPosition(temporalImageDataDeque.at(0));
	while (int lastGoodFrameId = mainCycle(
		mediaInputStruct, calibrationMatrix, dataProcessingConditions,
		temporalImageDataDeque, newGlobalData) > 0
	) {
		std::deque<TemporalImageData> oldTempImageData = temporalImageDataDeque;
		temporalImageDataDeque.clear();
		temporalImageDataDeque.resize(OPTIMAL_DEQUE_SIZE);
		temporalImageDataDeque.at(0).rotation = oldTempImageData.at(lastGoodFrameId).rotation;
		temporalImageDataDeque.at(0).motion = oldTempImageData.at(lastGoodFrameId).motion;

		transferGlobalData(globalDataStruct, newGlobalData);
		newGlobalData.cameraRotations.clear();
		newGlobalData.spatialCameraPositions.clear();
		newGlobalData.spatialPoints.clear();
		newGlobalData.spatialPointsColors.clear();
	}

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