#include "cameraCalibration.h"
#include "photosProcessingCycle.h"
#include "videoProcessingCycle.h"
#include "IOmisc.h"
#include "ceres/ceres.h"

#include "main_config.h"

using namespace cv;

int main(int argc, char** argv) {
    google::InitGoogleLogging("BA");
#ifdef CALIB
    std::vector<String> files;
    glob("../static/for_calib/samsung-horizontal-p/*.JPG", files, false);
    chessboardPhotosCalibration(files, 14);
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