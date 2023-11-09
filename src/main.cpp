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

#define ESC_KEY 27
#define FEATURE_EXTRACTING_THRESHOLD 10
#define FEATURE_TRACKING_BARRIER 20
#define FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE 10000
using namespace cv;


#define FRAMES_GAP 10
#define REQUIRED_EXTRACTED_POINTS_COUNT 10

int main(int argc, char** argv)
{
	Mat currentFrame, previousFrame, result;
	std::vector<KeyPoint> currentFrameExtractedKeyPoints;
	std::vector<Point2f> currentFrameExtractedPoints;
	std::vector<Point2f> previousFrameExtractedPoints;
	std::vector<Point2f> currentFrameTrackedPoints;
	std::ofstream reportStream;
	char report[256];
	reportStream.open(report);
	VideoCapture cap("data/indoor_speed.mp4");
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}


	Mat previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
		currentProjectionMatrix(3, 4, CV_64F);

	Mat calibrationMatrix(3, 3, CV_64F);
	calibration(calibrationMatrix, CalibrationOption::load);

	int countOfFrames = 0;
	bool first = true;
	while (true) {
		cap.read(currentFrame);
		countOfFrames++;
		if (countOfFrames >= FRAMES_GAP) {
			countOfFrames = 0;
			cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
			cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
			fastExtractor(currentFrame, currentFrameExtractedKeyPoints, FEATURE_EXTRACTING_THRESHOLD);
			if (currentFrameExtractedKeyPoints.size() < REQUIRED_EXTRACTED_POINTS_COUNT)
				continue;
			if (first) {
				KeyPoint::convert(currentFrameExtractedKeyPoints, currentFrameExtractedPoints);
				previousFrameExtractedPoints = currentFrameExtractedPoints;
				previousFrame = currentFrame.clone();
				first = false;
				continue;
			}


			trackFeatures(previousFrameExtractedPoints, currentFrameTrackedPoints, previousFrame,
				currentFrame, FEATURE_TRACKING_BARRIER, FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE);
			Mat rotationMatrix = Mat::zeros(3, 3, CV_64F),
				translationVector = Mat::zeros(3, 1, CV_64F);
			if (estimateProjection(previousFrameExtractedPoints, currentFrameTrackedPoints, calibrationMatrix, rotationMatrix,
				translationVector, currentProjectionMatrix)) {
				Mat homogeneous3DPoints;
				triangulate(previousFrameExtractedPoints, currentFrameTrackedPoints, previousProjectionMatrix,
					currentProjectionMatrix, homogeneous3DPoints);
				previousFrameExtractedPoints = currentFrameExtractedPoints;
				previousProjectionMatrix = currentProjectionMatrix;
				previousFrame = currentFrame;
			}
			reportStream << "Features extracted: " << previousFrameExtractedPoints.size() << std::endl;
			reportStream << "Tracked points: " << currentFrameTrackedPoints.size() << std::endl;
			reportStream << "Current projection matrix:\n" << currentProjectionMatrix << std::endl << std::endl;

		}
		reportStream.flush();
		reportStream.close();
	}
	return 0;
}