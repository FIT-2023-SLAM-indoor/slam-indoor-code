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

static void setReportsPaths(
        char* path, std::ofstream& reportStream, std::ofstream& d3PointsStream
) {
    char tmp[256] = "";
    sprintf(tmp, "%s/main.txt", path);
    reportStream.open(tmp);
    sprintf(tmp, "%s/3Dpoints.txt", path);
    d3PointsStream.open(tmp);
}

#define ESC_KEY 27
int videoProcessingCycle(VideoCapture& cap, int featureTrackingBarier, int featureTrackingMaxAcceptableDiff,
	int framesGap, int requiredExtractedPointsCount, int featureExtractingThreshold, char* reportsDirPath)
{
	Mat currentFrame, previousFrame, result, homogeneous3DPoints;
	std::vector<KeyPoint> currentFrameExtractedKeyPoints;

	std::vector<Point2f> currentFrameExtractedPoints;
	std::vector<Point2f> previousFrameExtractedPoints;
	std::vector<Point2f> previousFrameExtractedPointsTemp;
	std::vector<Point2f> currentFrameTrackedPoints;
	std::ofstream reportStream;
    std::ofstream d3PointsStream;
    setReportsPaths(reportsDirPath, reportStream, d3PointsStream);


	Mat previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
		currentProjectionMatrix(3, 4, CV_64F),
        worldCameraPose = (Mat_<double>(1, 3) << 0, 0, 0),
        worldCameraRotation = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

	Mat calibrationMatrix(3, 3, CV_64F);
	calibration(calibrationMatrix, CalibrationOption::load);

	int countOfFrames = 0;
	bool first = true;
	while (true) {
		cap.read(currentFrame);
		countOfFrames++;
		if (countOfFrames < framesGap)
			continue;
		cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
		cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
		fastExtractor(currentFrame, currentFrameExtractedKeyPoints, featureExtractingThreshold);
		if (currentFrameExtractedKeyPoints.size() < requiredExtractedPointsCount)
			continue;
		KeyPoint::convert(currentFrameExtractedKeyPoints, currentFrameExtractedPoints);
		if (first) {

			previousFrameExtractedPoints = currentFrameExtractedPoints;
			previousFrame = currentFrame.clone();
			first = false;
			continue;
		}

		reportStream << countOfFrames << std::endl;
		reportStream << "prev features extracted: " << previousFrameExtractedPoints.size() << std::endl;
		reportStream << "curr features extracted: " << currentFrameExtractedPoints.size() << std::endl;
		previousFrameExtractedPointsTemp = previousFrameExtractedPoints;
		trackFeatures(previousFrameExtractedPointsTemp, previousFrame,
			currentFrame, currentFrameTrackedPoints, featureTrackingBarier, featureTrackingMaxAcceptableDiff);

		reportStream << "changed feat extracted: " << previousFrameExtractedPointsTemp.size() << std::endl;
		reportStream << "Tracked points: " << currentFrameTrackedPoints.size() << std::endl;
		if (currentFrameTrackedPoints.size() < requiredExtractedPointsCount) {
			currentFrameTrackedPoints.clear();
			currentFrameExtractedPoints.clear();
			continue;
		}

        Mat previousFrameExtractedPointsMatrix = Mat(previousFrameExtractedPointsTemp);
        Mat currentFrameTrackedPointsMatrix = Mat(currentFrameTrackedPoints);
        previousFrameExtractedPointsMatrix.reshape(1).convertTo(previousFrameExtractedPointsMatrix, CV_64F);
        currentFrameTrackedPointsMatrix.reshape(1).convertTo(currentFrameTrackedPointsMatrix, CV_64F);
		Mat rotationMatrix = Mat::zeros(3, 3, CV_64F),
			translationVector = Mat::zeros(3, 1, CV_64F);
		if (estimateProjection(previousFrameExtractedPointsMatrix,
            currentFrameTrackedPointsMatrix, calibrationMatrix, rotationMatrix,
            translationVector, currentProjectionMatrix)) {

			triangulate(previousFrameExtractedPointsMatrix,
                        currentFrameTrackedPointsMatrix, previousProjectionMatrix,
				currentProjectionMatrix, homogeneous3DPoints);
            reportStream << "Homogeneous 3D points: " << homogeneous3DPoints.cols << std::endl;
            Mat euclideanPoints;
            convertPointsFromHomogeneousWrapper(homogeneous3DPoints, euclideanPoints);
            reportStream << "3D points: " << euclideanPoints.rows << std::endl << std::endl;
            Mat worldEuclideanPoints = euclideanPoints.clone();
            placeEuclideanPointsInWorldSystem(worldEuclideanPoints, worldCameraPose, worldCameraRotation);
            d3PointsStream << "3D points in world system: " << worldEuclideanPoints.rows << std::endl
                           << worldEuclideanPoints << std::endl << std::endl;

            refineWorldCameraPose(rotationMatrix, translationVector, worldCameraPose, worldCameraRotation);
            reportStream << "New world camera pose: " << worldCameraPose << std::endl << std::endl;
            reportStream << "New world camera rotation: " << worldCameraRotation << std::endl << std::endl;

			previousFrameExtractedPoints = currentFrameExtractedPoints;

			previousProjectionMatrix = currentProjectionMatrix.clone();
			previousFrame = currentFrame.clone();
		}
		currentFrameTrackedPoints.clear();
		currentFrameExtractedPoints.clear();
		reportStream << "Current projection matrix:\n" << currentProjectionMatrix << std::endl << std::endl;
		reportStream.flush();
        d3PointsStream.flush();
		countOfFrames = 0;
		char c = (char)waitKey(1000);

		if (c == ESC_KEY)
			break;
	}

	reportStream.close();
    d3PointsStream.close();
	return 0;
}