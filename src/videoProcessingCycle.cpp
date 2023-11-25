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
	int framesBatchSize, int requiredExtractedPointsCount, int featureExtractingThreshold, char* reportsDirPath)
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
	std::vector<Mat> batch;
	std::vector<Mat> newBatch;
	int findedIndex = 0;
	int countOfFrames = 0;
	bool first = true;
	while (cap.read(currentFrame)) {

		fastExtractor(currentFrame, currentFrameExtractedKeyPoints, featureExtractingThreshold);
		if (currentFrameExtractedKeyPoints.size() < requiredExtractedPointsCount)
			continue;
		if (first) {
			KeyPoint::convert(currentFrameExtractedKeyPoints, currentFrameExtractedPoints);
			cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
			cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
			previousFrameExtractedPoints = currentFrameExtractedPoints;
			previousFrame = currentFrame.clone();
			first = false;
			currentFrameExtractedPoints.clear();
			currentFrameExtractedKeyPoints.clear();
			continue;
		}

		countOfFrames++;
		if (countOfFrames <= framesBatchSize) {
			batch.push_back(currentFrame.clone());
			continue;
		}



		reportStream << "prev features extracted: " << previousFrameExtractedPoints.size() << std::endl;
		int findIndex = -1;

		for (int batchIndex = batch.size() - 1;batchIndex >= 0;batchIndex--) {
			currentFrame = batch.at(batchIndex);
			cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
			cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
			previousFrameExtractedPointsTemp = previousFrameExtractedPoints;
			trackFeatures(previousFrameExtractedPointsTemp, previousFrame,
				currentFrame, currentFrameTrackedPoints, featureTrackingBarier, featureTrackingMaxAcceptableDiff);
			if (currentFrameTrackedPoints.size() < requiredExtractedPointsCount) {
				reportStream << "currentFrameTrackedPoints:" << currentFrameTrackedPoints.size() << std::endl;
				currentFrameTrackedPoints.clear();
				continue;
			}
			else {
				findIndex = batchIndex;
				reportStream << batchIndex << std::endl;
				previousFrame = currentFrame.clone();
				fastExtractor(currentFrame, currentFrameExtractedKeyPoints, featureExtractingThreshold);
				KeyPoint::convert(currentFrameExtractedKeyPoints, currentFrameExtractedPoints);
				previousFrameExtractedPoints = currentFrameExtractedPoints;
				break;
			}

		}
		if (findIndex != -1) {
			for (int i = findIndex + 1;i < batch.size();i++)
				newBatch.push_back(batch.at(i));
		}
		else {
			batch.clear();
			reportStream << "Batch skipped" << std::endl;
			newBatch.clear();
			first = 1;
			countOfFrames = 0;
			previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
			currentFrameTrackedPoints.clear();
			currentFrameExtractedPoints.clear();
			previousFrameExtractedPoints.clear();
			currentFrameExtractedPoints.clear();
			continue;
		}

		reportStream << "changed feat extracted: " << previousFrameExtractedPointsTemp.size() << std::endl;
		reportStream << "Tracked points: " << currentFrameTrackedPoints.size() << std::endl;


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
			d3PointsStream << worldEuclideanPoints << std::endl << std::endl;
			/*d3PointsStream << "3D points in world system: " << worldEuclideanPoints.rows << std::endl
				<< worldEuclideanPoints << std::endl << std::endl;*/

			refineWorldCameraPose(rotationMatrix, translationVector, worldCameraPose, worldCameraRotation);
			reportStream << "New world camera pose: " << worldCameraPose << std::endl << std::endl;
			reportStream << "New world camera rotation: " << worldCameraRotation << std::endl << std::endl;
			previousProjectionMatrix = currentProjectionMatrix.clone();

		}


		reportStream << "Current projection matrix:\n" << currentProjectionMatrix << std::endl << std::endl;

#ifdef SHOW_TRACKED_POINTS
		Mat pointFrame = currentFrame.clone();
		for (int i = 0;i < currentFrameTrackedPoints.size();i++) {
			Vec3b& color = pointFrame.at<Vec3b>(currentFrameTrackedPoints.at(i));;
			color[0] = 0;
			color[1] = 0;
			color[2] = 255;
			pointFrame.at<Vec3b>(currentFrameTrackedPoints.at(i)) = color;
		}
		imshow("dd", pointFrame);
		waitKey(1000);
#endif
		reportStream.flush();
		d3PointsStream.flush();
		countOfFrames = newBatch.size();
		currentFrameTrackedPoints.clear();
		currentFrameExtractedPoints.clear();
		currentFrameExtractedKeyPoints.clear();
		previousFrameExtractedPointsTemp.clear();
		batch.clear();
		batch = newBatch;
		newBatch.clear();

	}

	reportStream.close();
	d3PointsStream.close();
	return 0;
}