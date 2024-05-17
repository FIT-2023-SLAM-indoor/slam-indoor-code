#include "fstream"
#include "ceres/ceres.h"

#include "config/config.h"
#include "misc/ChronoTimer.h"
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

#ifdef USE_CUDA
	if (cuda::getCudaEnabledDeviceCount() < 1) {
		std::cerr << "There's no available CUDA devices" << std::endl;
		return 3;
	}
	else {
		std::cout << "CUDA devices: " << cuda::getCudaEnabledDeviceCount() << std::endl;
	}
#endif

	ChronoTimer timer;

	if (argc < 2) {
		std::cerr << "Please specify path to JSON-config as the second argument" << std::endl;
		return 2;
	}

	configService.setConfigFile(argv[1]);
	google::InitGoogleLogging("BA");
	openLogsStreams();

	if (configService.getValue<bool>(ConfigFieldEnum::CALIBRATE)) {
		mainCalibrationEntryPoint();
		return 0;
	}

	std::string path = configService.getValue<std::string>(ConfigFieldEnum::OUTPUT_DATA_DIR);

	MediaSources mediaInputStruct;
	DataProcessingConditions dataProcessingConditions;
	Mat calibrationMatrix;
	defineProcessingEnvironment(mediaInputStruct, dataProcessingConditions, calibrationMatrix);
	
	GlobalData globalDataStruct;
	std::deque<TemporalImageData> oldTempImageDataDeque;
	int lastFrameOfLaunchId = -1;
	do {
		std::deque<TemporalImageData> newTempImageDataDeque(OPTIMAL_DEQUE_SIZE);
		defineCameraPosition(oldTempImageDataDeque, lastFrameOfLaunchId,
						     newTempImageDataDeque.at(0));

		GlobalData newGlobalData;
		lastFrameOfLaunchId = mainCycle(mediaInputStruct, calibrationMatrix,
										dataProcessingConditions, newTempImageDataDeque,
										newGlobalData);
		oldTempImageDataDeque = newTempImageDataDeque;

		insertNewGlobalData(globalDataStruct, newGlobalData);
	} while (lastFrameOfLaunchId > 0);

	rawOutput(globalDataStruct.spatialPoints, logStreams.pointStream);
	logStreams.pointStream.flush();
	rawOutput(globalDataStruct.spatialPointsColors, logStreams.colorStream);
	logStreams.colorStream.flush();

	checkGlobalDataStruct(globalDataStruct);

	printDivider(logStreams.timeStream);
	timer.printStartDelta("Whole time: ", logStreams.timeStream);

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
