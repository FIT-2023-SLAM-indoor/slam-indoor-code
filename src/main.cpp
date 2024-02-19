#include "cameraCalibration.h"
#include "photosProcessingCycle.h"
#include "videoProcessingCycle.h"

#include "main_config.h"

using namespace cv;

int main(int argc, char** argv)
{
#ifdef CALIB
    std::vector<String> files;
    glob("./docs/artifact/calibration/for_calib_1/*.JPG", files, false);
    chessboardPhotosCalibration(files, 11);
    return 0;
#endif
#ifdef PHOTOS_CYCLE
    std::vector<String> photos;
    glob(PHOTOS_PATH_PATTERN, photos, false);
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
                         FEATURE_TRACKING_BARRIER,
                         FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE,
                         FRAMES_BATCH_SIZE,
                         REQUIRED_EXTRACTED_POINTS_COUNT,
                         FEATURE_EXTRACTING_THRESHOLD,
                         path);
#endif
    return 0;
}