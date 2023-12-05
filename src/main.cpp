#include <iostream>

#include "cameraCalibration.h"
#include "reportCycleForTwoFramesPair.h"
#include "videoProcessingCycle.h"
#include "main_config.h"
#include "cameraTransition.h"

using namespace cv;

#define NDEBUG
#define NCALIB

int main(int argc, char** argv)
{
#ifdef CALIB
    std::vector<String> files;
    glob("./data/for_calib/pixel/*.jpg", files, false);
    chessboardPhotosCalibration(files, 15);
    return 0;
#endif
#ifdef DEBUG
    reportingCycleForFramesPairs(
            FEATURE_EXTRACTING_THRESHOLD,
            FEATURE_TRACKING_BARRIER,
            FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE
    );
#else
    VideoCapture cap(VIDEO_SOURCE_PATH);
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}
	char path[] = OUTPUT_DATA_DIR;
	videoProcessingCycle(cap,
                         FEATURE_TRACKING_BARRIER,
                         FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE,
                         FRAMES_BATCH_SIZE,
                         REQUIRED_EXTRACTED_POINTS_COUNT,
                         FEATURE_EXTRACTING_THRESHOLD,
                         path);
#endif
    return 0;
}