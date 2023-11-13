#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "fastExtractor.h"
#include "featureTracking.h"
#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "triangulate.h"
#include "videoProcessingCycle.h"
#define ESC_KEY 27
#define FEATURE_EXTRACTING_THRESHOLD 25
#define FEATURE_TRACKING_BARRIER 20
#define FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE 10000
using namespace cv;


#define FRAMES_GAP 2
#define REQUIRED_EXTRACTED_POINTS_COUNT 10

int main(int argc, char** argv)
{
	VideoCapture cap("data/indoor_test.mp4");
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}
	videoProcessingCycle(cap, 10, 10000, 3, 10, 20);
	return 0;
}