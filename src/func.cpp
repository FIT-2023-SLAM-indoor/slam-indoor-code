#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>
#include <iostream>

#include "fastExtractor.h"
#include "featureTracking.h"
#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "triangulate.h"


int mainCycle(const int FEATURE_EXTRACTING_THRESHOLD, const int FEATURE_TRACKING_BARRIER,
	const int FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE) {
	Mat image, image2, result;
	std::vector<KeyPoint> keypoints;


	VideoCapture cap("data/indoor_speed.mp4");
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}

	Mat previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
		currentProjectionMatrix(3, 4, CV_64F);

	Mat calibrationMatrix(3, 3, CV_64F);
	calibration(calibrationMatrix, CalibrationOption::load);

	while (true) {
		cap.read(image);
		cvtColor(image, image, COLOR_BGR2GRAY);
		cvtColor(image, image, COLOR_GRAY2BGR);

		// Applied the FAST algorithm to the image and saved the image
		// with the highlighted features in @result
		fastExtractor(image, keypoints, threshold);
		drawKeypoints(image, keypoints, result);

		namedWindow("Display Image", WINDOW_AUTOSIZE);
		imshow("Display Image", result);
		waitKey(1000);

		//Transform image into black and white
		cap.read(image2);
		cvtColor(image2, image2, COLOR_BGR2GRAY);
		cvtColor(image2, image2, COLOR_GRAY2BGR);

		std::vector<Point2f> features;
		std::vector<Point2f> newFeatures;
		KeyPoint::convert(keypoints, features);

		trackFeatures(features, image, image2, newFeatures, featureTrackingBarier, 10000);

		//Getting keypoints vector to show from points vector(needed only for afcts, you can delete it)
		std::vector<KeyPoint> keyPoints2;
		KeyPoint::convert(newFeatures, keyPoints2);
		drawKeypoints(image2, keyPoints2, result);
		imshow("Display Image", result);

		Mat q = Mat(features);
		Mat g = Mat(newFeatures);
		q = q.reshape(1);
		g = g.reshape(1);

		////////////////////////////////////////
		// Estimate matrices
		////////////////////////////////////////
		currentProjectionMatrix = Mat(3, 4, CV_64F);
		Mat rotationMatrix = Mat::zeros(3, 3, CV_64F),
			translationVector = Mat::zeros(3, 1, CV_64F);
		estimateProjection(q, g, calibrationMatrix, rotationMatrix,
			translationVector, currentProjectionMatrix);
		////////////////////////////////////////

		Mat homogeneous3DPoints;
		triangulate(q, g, previousProjectionMatrix,
			currentProjectionMatrix, homogeneous3DPoints);
		previousProjectionMatrix = currentProjectionMatrix;

		//////////////////////////////////////

		char c = (char)waitKey(1000);
		Vec3f res = rotationMatrixToEulerAngles(rotationMatrix);
		std::cout << res << std::endl;
		if (c == ESC_KEY)
			break;
	}
	return 0;
}