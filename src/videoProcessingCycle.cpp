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
	char* path, std::ofstream& reportStream, std::ofstream& d3PointsStream,
    std::ofstream& d3PointsStream1, std::ofstream& d3PointsStream2, std::ofstream& d3PointsStream3,
    std::ofstream& d3PointsStream4,
    std::ofstream& poseStream, std::ofstream& poseStream2
) {
	char tmp[256] = "";
	sprintf(tmp, "%s/main.txt", path);
	reportStream.open(tmp);
	sprintf(tmp, "%s/3DpointsTriangMlt.txt", path);
	d3PointsStream.open(tmp);
    sprintf(tmp, "%s/3DpointsRecoverMlt.txt", path);
    d3PointsStream1.open(tmp);
    sprintf(tmp, "%s/3DpointsTriangRt.txt", path);
    d3PointsStream2.open(tmp);
    sprintf(tmp, "%s/3DpointsRecoverRt.txt", path);
    d3PointsStream3.open(tmp);
    sprintf(tmp, "%s/3DpointsGloablTriang.txt", path);
    d3PointsStream4.open(tmp);
    sprintf(tmp, "%s/pose.txt", path);
    poseStream.open(tmp);
    sprintf(tmp, "%s/pose_handy_calc.txt", path);
    poseStream2.open(tmp);
}

#define SHOW_TRACKED_POINTS
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
    std::ofstream d3PointsStream1;
    std::ofstream d3PointsStream2;
    std::ofstream d3PointsStream3;
    std::ofstream d3PointsStream4;
    std::ofstream poseStream;
    std::ofstream poseStream2;
	setReportsPaths(reportsDirPath, reportStream,
                    d3PointsStream, d3PointsStream1, d3PointsStream2, d3PointsStream3, d3PointsStream4,
                    poseStream, poseStream2);


	Mat originProjection = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
		currentProjectionMatrix(3, 4, CV_64F),
		worldCameraPose = (Mat_<double>(3, 1) << 0, 0, 0),
        worldCameraPoseFromHandCalc = (Mat_<double>(3, 1) << 0, 0, 0),
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
//			previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
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
			translationVector = Mat::zeros(3, 1, CV_64F),
            triangulatedPointsFromRecoverPose;
		if (estimateProjection(previousFrameExtractedPointsMatrix,
                               currentFrameTrackedPointsMatrix, calibrationMatrix, rotationMatrix,
                               translationVector, currentProjectionMatrix, triangulatedPointsFromRecoverPose)) {

            triangulate(previousFrameExtractedPointsMatrix,
                        currentFrameTrackedPointsMatrix, originProjection,
                        currentProjectionMatrix, homogeneous3DPoints);
            reportStream << "3D points count: " << homogeneous3DPoints.cols << std::endl;
            Mat normalizedHomogeneous3DPointsFromTriangulation, normalizedHomogeneous3DPointsFromRecoverPose;
            normalizeHomogeneousWrapper(homogeneous3DPoints, normalizedHomogeneous3DPointsFromTriangulation);
            normalizeHomogeneousWrapper(triangulatedPointsFromRecoverPose, normalizedHomogeneous3DPointsFromRecoverPose);

            Mat newGlobalProjectionMatrix(4, 4, CV_64F);
            addHomogeneousRow(previousProjectionMatrix);
            addHomogeneousRow(currentProjectionMatrix);

            Mat H3DPointsFromTriangulationInWorldUsingP = normalizedHomogeneous3DPointsFromTriangulation.clone();
            H3DPointsFromTriangulationInWorldUsingP = previousProjectionMatrix * H3DPointsFromTriangulationInWorldUsingP;
            Mat H3DPointsFromRecoverPoseInWorldUsingP = normalizedHomogeneous3DPointsFromTriangulation.clone();
            H3DPointsFromRecoverPoseInWorldUsingP = previousProjectionMatrix * H3DPointsFromRecoverPoseInWorldUsingP;

            newGlobalProjectionMatrix = previousProjectionMatrix * currentProjectionMatrix;

            removeHomogeneousRow(newGlobalProjectionMatrix);
            removeHomogeneousRow(previousProjectionMatrix);

            addHomogeneousRow(worldCameraPose);
            worldCameraPose = currentProjectionMatrix * worldCameraPose;
            removeHomogeneousRow(worldCameraPose);
            removeHomogeneousRow(currentProjectionMatrix);


			Mat worldEuclideanPoints(3, homogeneous3DPoints.cols, CV_64F);
            worldEuclideanPoints = H3DPointsFromTriangulationInWorldUsingP.rowRange(0, 3).clone();
            Mat worldEuclideanPointsFromRecoverPose(3, homogeneous3DPoints.cols, CV_64F);
            worldEuclideanPointsFromRecoverPose = H3DPointsFromRecoverPoseInWorldUsingP.rowRange(0, 3).clone();
			d3PointsStream << worldEuclideanPoints.t() << std::endl << std::endl;
            d3PointsStream1 << worldEuclideanPointsFromRecoverPose.t() << std::endl << std::endl;

            reportStream << "Current projection: " << currentProjectionMatrix << std::endl << std::endl;
			reportStream << "New world camera pose from multiply: " << worldCameraPose << std::endl << std::endl;
            poseStream << worldCameraPose.t() << std::endl << std::endl;
			reportStream << "New world camera projection: " << newGlobalProjectionMatrix << std::endl << std::endl;

            // Part with WORLD triangulation
            Mat globalTriangulatedHomogeneousPoints;
            triangulate(previousFrameExtractedPointsMatrix,
                        currentFrameTrackedPointsMatrix, previousProjectionMatrix,
                        newGlobalProjectionMatrix, globalTriangulatedHomogeneousPoints);
            Mat normalizedGlobalTriangulatedHomogeneousPoints;
            normalizeHomogeneousWrapper(globalTriangulatedHomogeneousPoints, normalizedGlobalTriangulatedHomogeneousPoints);
            Mat euclideanGlobalTriangulatedHomogeneousPoints = normalizedGlobalTriangulatedHomogeneousPoints.rowRange(0, 3).clone();
            d3PointsStream4 << euclideanGlobalTriangulatedHomogeneousPoints.t() << std::endl << std::endl;

            // Part with old-concept global pose estimating
            Mat euclidean3DPointsFromTriangulationInWorldUsingRt = normalizedHomogeneous3DPointsFromTriangulation.rowRange(0, 3).clone();
            placeEuclideanPointsInWorldSystem(euclidean3DPointsFromTriangulationInWorldUsingRt, worldCameraPoseFromHandCalc, worldCameraRotation);
            Mat euclidean3DPointsFromRecoverPoseInWorldUsingRt = normalizedHomogeneous3DPointsFromTriangulation.rowRange(0, 3).clone();
            placeEuclideanPointsInWorldSystem(euclidean3DPointsFromRecoverPoseInWorldUsingRt, worldCameraPoseFromHandCalc, worldCameraRotation);

            refineWorldCameraPose(rotationMatrix, translationVector, worldCameraPoseFromHandCalc, worldCameraRotation);

            d3PointsStream2 << euclidean3DPointsFromTriangulationInWorldUsingRt.t() << std::endl << std::endl;
            d3PointsStream3 << euclidean3DPointsFromRecoverPoseInWorldUsingRt.t() << std::endl << std::endl;
            reportStream << "New world camera pose from handy calc: " << worldCameraPoseFromHandCalc << std::endl << std::endl;
            reportStream << "New world camera rotation from handy calc: " << worldCameraRotation << std::endl << std::endl;
            poseStream2 << worldCameraPoseFromHandCalc.t() << std::endl << std::endl;


            previousProjectionMatrix = newGlobalProjectionMatrix.clone();
		}

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
//        resizeWindow("dd", pointFrame.cols/4, pointFrame.rows/4);
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
    d3PointsStream1.close();
    d3PointsStream2.close();
    d3PointsStream3.close();
    d3PointsStream4.close();
    poseStream.close();
    poseStream2.close();
	return 0;
}