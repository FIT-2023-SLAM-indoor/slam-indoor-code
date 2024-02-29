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
#include "featureMatching.h"
#include "triangulate.h"

#include "photosProcessingCycle.h"


static void setReportsPaths(
        char* path, std::ofstream& reportStream,
        std::ofstream& d3PointsStream,
        std::ofstream& poseStream,
        std::ofstream& poseStreamHandy,
        std::ofstream& poseTestStream
) {
    char tmp[256] = "";
    sprintf(tmp, "%s/main.txt", path);
    reportStream.open(tmp);
    sprintf(tmp, "%s/points.txt", path);
    d3PointsStream.open(tmp);
    sprintf(tmp, "%s/pose.txt", path);
    poseStream.open(tmp);
    sprintf(tmp, "%s/pose_hand.txt", path);
    poseStreamHandy.open(tmp);
    sprintf(tmp, "%s/pose_test.txt", path);
    poseTestStream.open(tmp);
}

int photosProcessingCycle(std::vector<String> &photosPaths, int featureTrackingBarier, int featureTrackingMaxAcceptableDiff,
                         int framesBatchSize, int requiredExtractedPointsCount, int featureExtractingThreshold, char* reportsDirPath)
{
    Mat preCurrentFrame, currentFrame, previousFrame, result, homogeneous3DPoints;
    std::vector<KeyPoint> currentFrameExtractedKeyPoints;
    std::vector<KeyPoint> previousFrameExtractedKeyPoints;
    std::vector<Point2f> currentFrameExtractedPoints;
    std::vector<Point2f> previousFrameExtractedPoints;
    std::vector<Point2f> previousFrameExtractedPointsTemp;
    std::vector<Point2f> currentFrameTrackedPoints;

    std::ofstream reportStream;
    std::ofstream pointsStream;
    std::ofstream poseStream;
    std::ofstream poseHandyStream;
    std::ofstream poseTestStream;
    setReportsPaths(reportsDirPath, reportStream, pointsStream, poseStream, poseHandyStream, poseTestStream);

    Mat originProjection = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        currentProjectionMatrix(3, 4, CV_64F),
        worldCameraPose = (Mat_<double>(3, 1) << 0, 0, 0),
        worldCameraPoseFromHandCalc = (Mat_<double>(3, 1) << 0, 0, 0),
        worldCameraRotation = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    Mat calibrationMatrix(3, 3, CV_64F), distCoeffs(1, 5, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);
    loadMatrixFromXML(CALIBRATION_PATH, distCoeffs, "DC");

    std::vector<Mat> batch;
    std::vector<Mat> newBatch;
    int findedIndex = 0;
    int countOfFrames = 0;
    bool first = true;
    for (auto photoPath : photosPaths) {
#ifdef USE_UNDISTORTION
        preCurrentFrame = imread(photoPath);
        // PART FOR UNDISTROTION
        undistort(preCurrentFrame, currentFrame, calibrationMatrix, distCoeffs);
        ////////////////////////
#else
        currentFrame = imread(photoPath);
#endif
        fastExtractor(currentFrame, currentFrameExtractedKeyPoints, featureExtractingThreshold);
        if (currentFrameExtractedKeyPoints.size() < requiredExtractedPointsCount)
            continue;
        if (first) {
            KeyPoint::convert(currentFrameExtractedKeyPoints, currentFrameExtractedPoints);
#ifdef FT_ACTIVATE
            cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
            cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
#endif
            previousFrameExtractedPoints = currentFrameExtractedPoints;
            previousFrameExtractedKeyPoints = currentFrameExtractedKeyPoints;
            previousFrame = currentFrame.clone();
            first = false;
            currentFrameExtractedPoints.clear();
            currentFrameExtractedKeyPoints.clear();
            continue;
        }



        if (countOfFrames < framesBatchSize) {
            batch.push_back(currentFrame.clone());
            countOfFrames++;
            if (countOfFrames < framesBatchSize)
                continue;
        }
        reportStream << "prev features extracted: " << previousFrameExtractedPoints.size() << std::endl;
        int findIndex = -1;

        for (int batchIndex = batch.size() - 1;batchIndex >= 0;batchIndex--) {
            currentFrame = batch.at(batchIndex);
            previousFrameExtractedPointsTemp = previousFrameExtractedPoints;
#ifdef FT_ACTIVATE
            cvtColor(currentFrame, currentFrame, COLOR_BGR2GRAY);
            cvtColor(currentFrame, currentFrame, COLOR_GRAY2BGR);
			trackFeatures(previousFrameExtractedPointsTemp, previousFrame,
				currentFrame, currentFrameTrackedPoints, featureTrackingBarier, featureTrackingMaxAcceptableDiff);
#else
            previousFrameExtractedPointsTemp.clear();
            featureMatching(previousFrame, currentFrame, previousFrameExtractedKeyPoints, currentFrameExtractedKeyPoints,
                currentFrameTrackedPoints, previousFrameExtractedPointsTemp);
#endif
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
            waitKey(1000);
#endif
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
                previousFrameExtractedKeyPoints = currentFrameExtractedKeyPoints;
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
            currentFrameExtractedKeyPoints.clear();
            previousFrameExtractedKeyPoints.clear();
            previousFrameExtractedPoints.clear();
            currentFrameExtractedPoints.clear();
            continue;
        }

        reportStream << "changed feat extracted: " << previousFrameExtractedPointsTemp.size() << std::endl;
        reportStream << "Tracked points: " << currentFrameTrackedPoints.size() << std::endl;



        Mat rotationMatrix = Mat::zeros(3, 3, CV_64F),
                translationVector = Mat::zeros(3, 1, CV_64F),
                triangulatedPointsFromRecoverPose;

        if (estimateProjection(previousFrameExtractedPointsTemp,
                               currentFrameTrackedPoints, calibrationMatrix, rotationMatrix,
                               translationVector, currentProjectionMatrix, triangulatedPointsFromRecoverPose)) {

            Mat previousFrameExtractedPointsMatrix = Mat(previousFrameExtractedPointsTemp);
            Mat currentFrameTrackedPointsMatrix = Mat(currentFrameTrackedPoints);
            previousFrameExtractedPointsMatrix.reshape(1).convertTo(previousFrameExtractedPointsMatrix, CV_64F);
            currentFrameTrackedPointsMatrix.reshape(1).convertTo(currentFrameTrackedPointsMatrix, CV_64F);

            Mat newGlobalProjectionMatrix(4, 4, CV_64F);
            addHomogeneousRow(previousProjectionMatrix);
            addHomogeneousRow(currentProjectionMatrix);
            newGlobalProjectionMatrix = previousProjectionMatrix * currentProjectionMatrix;
            removeHomogeneousRow(newGlobalProjectionMatrix);
            removeHomogeneousRow(previousProjectionMatrix);

            triangulate(previousFrameExtractedPointsMatrix,
                        currentFrameTrackedPointsMatrix, calibrationMatrix * previousProjectionMatrix,
                        calibrationMatrix * newGlobalProjectionMatrix, homogeneous3DPoints);

            reportStream << "3D points count: " << homogeneous3DPoints.cols << std::endl;
            Mat normalizedHomogeneous3DPointsFromTriangulation;
            normalizeHomogeneousWrapper(homogeneous3DPoints, normalizedHomogeneous3DPointsFromTriangulation);
            Mat euclidean3DPointsFromTriangulationInWorldUsingRt = normalizedHomogeneous3DPointsFromTriangulation.rowRange(0, 3).clone();

            addHomogeneousRow(worldCameraPose);
            worldCameraPose = currentProjectionMatrix * worldCameraPose;
            removeHomogeneousRow(worldCameraPose);
            removeHomogeneousRow(currentProjectionMatrix);


            reportStream << "Current projection: " << currentProjectionMatrix << std::endl << std::endl;
            reportStream << "New world camera pose from multiply: " << worldCameraPose << std::endl << std::endl;
            poseStream << worldCameraPose.t() << std::endl << std::endl;
            reportStream << "New world camera projection: " << newGlobalProjectionMatrix << std::endl << std::endl;

            refineWorldCameraPose(rotationMatrix, translationVector, worldCameraPoseFromHandCalc, worldCameraRotation);

            pointsStream << euclidean3DPointsFromTriangulationInWorldUsingRt.t() << std::endl << std::endl;
            reportStream << "New world camera pose from handy calc: " << worldCameraPoseFromHandCalc << std::endl << std::endl;
            poseHandyStream << worldCameraPoseFromHandCalc.t() << std::endl << std::endl;
            reportStream << "New world camera rotation from handy calc: " << worldCameraRotation << std::endl << std::endl;

            Mat zeroPOose = (Mat_<double>(4, 1) << 0, 0, 0, 1);
            poseTestStream << (newGlobalProjectionMatrix * zeroPOose).t() << std::endl << std::endl;

            previousProjectionMatrix = newGlobalProjectionMatrix.clone();
        }
        reportStream.flush();
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
    pointsStream.close();
    poseStream.close();
    return 0;
}