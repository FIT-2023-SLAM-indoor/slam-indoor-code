#include <iostream>
#define _USE_MATH_DEFINES

#include "cameraCalibration.h"
#include "reportCycleForTwoFramesPair.h"
#include "videoProcessingCycle.h"

using namespace cv;

#define ESC_KEY 27
#define FEATURE_EXTRACTING_THRESHOLD 10
#define FEATURE_TRACKING_BARRIER 10
#define FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE 10000


#define FRAMES_GAP 2
#define REQUIRED_EXTRACTED_POINTS_COUNT 10

#define NDEBUG
#define NCALIB

int main(int argc, char** argv)
{
#ifdef CALIB
    std::vector<String> files;
    glob("./data/for_calib/roborock/merged1/*.JPG", files, false);
    chessboardPhotosCalibration(files, 20);
    return 0;
#endif
#ifdef DEBUG
    reportingCycleForFramesPairs(
            FEATURE_EXTRACTING_THRESHOLD,
            FEATURE_TRACKING_BARRIER,
            FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE
    );
#else
    VideoCapture cap("data/indoor_test.mp4");
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}
	char path[] = "./data/video_report";
	videoProcessingCycle(cap, 10, 10000, 3, 10, 20, path);
#endif
    return 0;
}