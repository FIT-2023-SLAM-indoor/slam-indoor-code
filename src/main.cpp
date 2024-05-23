#include "fstream"
#include "ceres/ceres.h"

#include "config/config.h"
#include "misc/ChronoTimer.h"
#include "misc/IOmisc.h"
#include "calibration/cameraCalibration.h"

#include "mainModule//cycleProcessing/mainCycle.h"
#include "mainModule/cycleProcessing/mainCycleInternals.h"
#include "vizualization/vizualizationModule.h"

using namespace cv;

const int OPTIMAL_DEQUE_SIZE = 8;

ConfigService configService;
LogFilesStreams logStreams;

/**
 * Function for main algorithm
 * @param mediaInputStruct
 * @param dataProcessingConditions
 * @return
 */
static GlobalData slamMain(Mat &calibrationMatrix);

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

	if (argc < 2) {
		std::cerr << "Please specify path to JSON-config as the second argument" << std::endl;
		return 2;
	}

	configService.setConfigFile(argv[1]);

	if (configService.getValue<bool>(ConfigFieldEnum::CALIBRATE)) {
		mainCalibrationEntryPoint();
		return 0;
	}

	Mat calibrationMatrix;
	defineCalibrationMatrix(calibrationMatrix);
	GlobalData globalDataStruct;
	if (configService.getValue<bool>(ConfigFieldEnum::ONLY_VIZ)) {
		globalDataStruct = getGlobalDataFromLogFiles();
	} else {
		globalDataStruct = slamMain(calibrationMatrix);
	}

	// Kostil for Points3f in visualizer
	std::vector<Point3f> convertedSpatialPoints;
	for (auto point : globalDataStruct.spatialPoints)
		convertedSpatialPoints.push_back(Point3f(point));


	vizualizePointsAndCameras(convertedSpatialPoints,
							  globalDataStruct.cameraRotations,
							  globalDataStruct.spatialCameraPositions,
							  globalDataStruct.spatialPointsColors,
							  calibrationMatrix);

    return 0;
}

static GlobalData slamMain(Mat &calibrationMatrix) {
	ChronoTimer timer;

	google::InitGoogleLogging("BA");
	openLogsStreams(logStreams);

	MediaSources mediaInputStruct;
	DataProcessingConditions dataProcessingConditions;
	defineProcessingEnvironment(mediaInputStruct, dataProcessingConditions);

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
	closeLogsStreams(logStreams);

	return globalDataStruct;
}
