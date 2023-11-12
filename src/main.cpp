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
#define FEATURE_EXTRACTING_THRESHOLD 25
#define FEATURE_TRACKING_BARRIER 20
#define FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE 10000
using namespace cv;


#define FRAMES_GAP 2
#define REQUIRED_EXTRACTED_POINTS_COUNT 10

int main(int argc, char** argv)
{
	Mat currentFrame, previousFrame, result, homogeneous3DPoints;
	std::vector<KeyPoint> currentFrameExtractedKeyPoints;

	std::vector<Point2f> currentFrameExtractedPoints;
	std::vector<Point2f> previousFrameExtractedPoints;
	std::vector<Point2f> previousFrameExtractedPointsTemp;
	std::vector<Point2f> currentFrameTrackedPoints;
	std::ofstream reportStream;
	char report[256];
	reportStream.open(report);
	VideoCapture cap("data/indoor_test.mp4");
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
		if (countOfFrames < FRAMES_GAP)
			continue;
		cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
		cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
		fastExtractor(currentFrame, currentFrameExtractedKeyPoints, FEATURE_EXTRACTING_THRESHOLD);
		if (currentFrameExtractedKeyPoints.size() < REQUIRED_EXTRACTED_POINTS_COUNT)
			continue;
		KeyPoint::convert(currentFrameExtractedKeyPoints, currentFrameExtractedPoints);
		if (first) {

			previousFrameExtractedPoints = currentFrameExtractedPoints;
			previousFrame = currentFrame.clone();
			first = false;
			continue;
		}

		std::cout << countOfFrames << std::endl;
		std::cout << "prev features extracted: " << previousFrameExtractedPoints.size() << std::endl;
		std::cout << "curr features extracted: " << currentFrameExtractedPoints.size() << std::endl;
		previousFrameExtractedPointsTemp = previousFrameExtractedPoints;
		trackFeatures(previousFrameExtractedPointsTemp, previousFrame,
			currentFrame, currentFrameTrackedPoints, FEATURE_TRACKING_BARRIER, FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE);

		std::cout << "changed feat extracted: " << previousFrameExtractedPointsTemp.size() << std::endl;
		std::cout << "Tracked points: " << currentFrameTrackedPoints.size() << std::endl;
		if (currentFrameTrackedPoints.size() < REQUIRED_EXTRACTED_POINTS_COUNT) {
			currentFrameTrackedPoints.clear();
			currentFrameExtractedPoints.clear();
			continue;
		}


		Mat rotationMatrix = Mat::zeros(3, 3, CV_64F),
			translationVector = Mat::zeros(3, 1, CV_64F);
		if (estimateProjection(Mat(previousFrameExtractedPointsTemp).reshape(1),
			Mat(currentFrameTrackedPoints).reshape(1), calibrationMatrix, rotationMatrix,
			translationVector, currentProjectionMatrix)) {

			triangulate(Mat(previousFrameExtractedPointsTemp).reshape(1),
				Mat(currentFrameTrackedPoints).reshape(1), previousProjectionMatrix,
				currentProjectionMatrix, homogeneous3DPoints);

			previousFrameExtractedPoints = currentFrameExtractedPoints;

			previousProjectionMatrix = currentProjectionMatrix.clone();
			previousFrame = currentFrame.clone();
		}
		currentFrameTrackedPoints.clear();
		currentFrameExtractedPoints.clear();
		reportStream << "Current projection matrix:\n" << currentProjectionMatrix << std::endl << std::endl;
		reportStream.flush();
		countOfFrames = 0;

	}


	reportStream.close();
	return 0;
}