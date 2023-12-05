//#include <cstdio>
//#include <iostream>
//#include <fstream>
//#include <iostream>
//#include <random>
//
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/calib3d.hpp>
//#include <opencv2/core/hal/hal.hpp>
//
//#include "cameraCalibration.h"
//#include "fastExtractor.h"
//#include "featureTracking.h"
//#include "cameraTransition.h"
//#include "triangulate.h"
//
//#include "reportCycleForTwoFramesPair.h"
//
//// Checks if a matrix is a valid rotation matrix.
//static bool isRotationMatrix(Mat& R)
//{
//    Mat Rt;
//    transpose(R, Rt);
//    Mat shouldBeIdentity = Rt * R;
//    Mat I = Mat::eye(3, 3, shouldBeIdentity.type());
//
//    return  norm(I, shouldBeIdentity) < 1e-6;
//
//}
//
//// Calculates rotation matrix to euler angles
//// The result is the same as MATLAB except the order
//// of the euler angles ( x and z are swapped ).
//static Vec3f rotationMatrixToEulerAngles(Mat& R)
//{
//
//    assert(isRotationMatrix(R));
//
//    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0)
//                    + R.at<double>(1, 0) * R.at<double>(1, 0));
//
//    bool singular = sy < 1e-6; // If
//
//    float x, y, z;
//    if (!singular)
//    {
//        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
//        y = atan2(-R.at<double>(2, 0), sy);
//        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
//    }
//    else
//    {
//        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
//        y = atan2(-R.at<double>(2, 0), sy);
//        z = 0;
//    }
//    return Vec3f(x, y, z);
//
//}
//
///**
// * Report's files paths generator.
// */
//static void setTwoFramesPaths(
//        int folderNumber, char* frame1, char* frame2, char* report, std::ofstream& reportStream,
//        std::ofstream& pointsStream1, std::ofstream& pointsStream2, std::ofstream& d3PointsStream
//) {
//    sprintf(frame1, "./data/two_frames/%d/0.png", folderNumber);
//    sprintf(frame2, "./data/two_frames/%d/1.png", folderNumber);
//    sprintf(report, "./data/two_frames/%d/report.txt", folderNumber);
//    reportStream.open(report);
//    char tmp[256] = "";
//    sprintf(tmp, "./data/two_frames/%d/points1.txt", folderNumber);
//    pointsStream1.open(tmp);
//    sprintf(tmp, "./data/two_frames/%d/points2.txt", folderNumber);
//    pointsStream2.open(tmp);
//    sprintf(tmp, "./data/two_frames/%d/3Dpoints.txt", folderNumber);
//    d3PointsStream.open(tmp);
//}
//
//#define ESC_KEY 27
//
//void reportingCycleForFramesPairs(const int FEATURE_EXTRACTING_THRESHOLD, const int FEATURE_TRACKING_BARRIER,
//                                  const int FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE) {
//    Mat image, image2, result;
//    std::vector<KeyPoint> featuresKeyPoints;
//
//    Mat previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
//            currentProjectionMatrix(3, 4, CV_64F),
//            worldCameraPose = (Mat_<double>(1, 3) << 0, 0, 0),
//            worldCameraRotation = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
//
//    Mat calibrationMatrix(3, 3, CV_64F);
//    calibration(calibrationMatrix, CalibrationOption::load);
//
//    char frame1[256], frame2[256], report[256];
//    std::ofstream reportStream, pointsStream1, pointsStream2, d3PointsStream;
//    for (int i = 1; i <= 8; ++i) {
//        try {
//            setTwoFramesPaths(i, frame1, frame2, report, reportStream, pointsStream1, pointsStream2, d3PointsStream);
//            std::cout << "Working with pair " << i << std::endl;
//            image = imread(frame1);
//            cvtColor(image, image, COLOR_BGR2GRAY);
//            cvtColor(image, image, COLOR_GRAY2BGR);
//
//
//            fastExtractor(image, featuresKeyPoints, FEATURE_EXTRACTING_THRESHOLD);
////            drawKeypoints(image, featuresKeyPoints, result);
////
////            namedWindow("Display Image", WINDOW_AUTOSIZE);
////            imshow("Display Image", result);
////            waitKey(1000);
//
//            //Transform image into black and white
//            image2 = imread(frame2);
//            cvtColor(image2, image2, COLOR_BGR2GRAY);
//            cvtColor(image2, image2, COLOR_GRAY2BGR);
//
//            std::vector<Point2f> featuresPoints;
//            KeyPoint::convert(featuresKeyPoints, featuresPoints);
//            reportStream << "Features extracted: " << featuresPoints.size() << std::endl
//                         << "Threshold: " << FEATURE_EXTRACTING_THRESHOLD << std::endl << std::endl;
////        std::cout << featuresPoints << std::endl;
//
//            std::vector<Point2f> trackedPoints;
//            trackFeatures(featuresPoints, image, image2, trackedPoints,
//                          FEATURE_TRACKING_BARRIER, FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE);
//            reportStream << "Tracked points: " << trackedPoints.size() << std::endl
//                         << "Barrier: " << FEATURE_TRACKING_BARRIER << std::endl
//                         << "Max acceptable difference: " << FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE << std::endl << std::endl;
//            pointsStream1 << "Features extracted and tracked: " << featuresPoints.size() << std::endl << std::endl
//                          << featuresPoints;
//            pointsStream2 << "Tracked points: " << trackedPoints.size() << std::endl
//                          << "Barrier: " << FEATURE_TRACKING_BARRIER << std::endl
//                          << "Max acceptable difference: " << FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE << std::endl << std::endl
//                          << trackedPoints;
////        std::cout << trackedPoints << std::endl;
//
//            //Getting keypoints vector to show from points vector(needed only for afcts, you can delete it)
////            std::vector<KeyPoint> trackedKeyPoints;
////            KeyPoint::convert(trackedPoints, trackedKeyPoints);
////            drawKeypoints(image2, trackedKeyPoints, result);
////            imshow("Display Image", result);
//
//            Mat q = Mat(featuresPoints);
//            Mat g = Mat(trackedPoints);
//            q.reshape(1).convertTo(q, CV_64F);
//            g.reshape(1).convertTo(g, CV_64F);
//
//            ////////////////////////////////////////
//            // Estimate matrices
//            ////////////////////////////////////////
//            currentProjectionMatrix = Mat(3, 4, CV_64F);
//            Mat rotationMatrix = Mat::zeros(3, 3, CV_64F),
//                    translationVector = Mat::zeros(3, 1, CV_64F);
//            estimateProjection(q, g, calibrationMatrix, rotationMatrix,
//                               translationVector, currentProjectionMatrix);
//            reportStream << "Current projection matrix:\n" << currentProjectionMatrix << std::endl << std::endl;
//            ////////////////////////////////////////
//
//            Mat homogeneous3DPoints;
//            triangulate(q, g, previousProjectionMatrix,
//                        currentProjectionMatrix, homogeneous3DPoints);
//            reportStream << "Homogeneous 3D points: " << homogeneous3DPoints.cols << std::endl;
//            Mat euclideanPoints;
//            convertPointsFromHomogeneousWrapper(homogeneous3DPoints, euclideanPoints);
//            reportStream << "3D points: " << euclideanPoints.rows << std::endl << std::endl;
//            Mat worldEuclideanPoints = euclideanPoints.clone();
//            placeEuclideanPointsInWorldSystem(worldEuclideanPoints, worldCameraPose, worldCameraRotation);
//            d3PointsStream << "3D points in world system: " << worldEuclideanPoints.rows << std::endl << std::endl
//                           << worldEuclideanPoints;
//
//            previousProjectionMatrix = currentProjectionMatrix;
//            refineWorldCameraPose(rotationMatrix, translationVector, worldCameraPose, worldCameraRotation);
//            reportStream << "New world camera pose: " << worldCameraPose << std::endl << std::endl;
//            reportStream << "New world camera rotation: " << worldCameraRotation << std::endl << std::endl;
//
//            char c = (char) waitKey(1000);
//            Vec3f res = rotationMatrixToEulerAngles(rotationMatrix);
//            reportStream << "Degrees rotations: " << res << std::endl << std::endl;
//            if (c == ESC_KEY)
//                break;
//        } catch (Exception& exception) {
//            reportStream << "Exception occurred: " << exception.what();
//            reportStream.flush();
//        }
//        reportStream.close();
//        pointsStream1.close();
//        pointsStream2.close();
//        d3PointsStream.close();
//    }
//}
