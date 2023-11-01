#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>

#include "fastExtractor.h"
#include "featureTracking.h"
#include "cameraCalibration.h"
#include "cameraTransition.h"
#include "triangulate.h"

#define ESC_KEY 27

using namespace cv;

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat& R)
{
	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return  norm(I, shouldBeIdentity) < 1e-6;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat& R)
{

	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return Vec3f(x, y, z);

}

int main(int argc, char** argv)
{
	Mat image, image2, result;
	std::vector<KeyPoint> keypoints;


//	VideoCapture cap("data/indoor_speed.mp4");
//	if (!cap.isOpened()) {
//		std::cerr << "Camera wasn't opened" << std::endl;
//		return -1;
//	}

	Mat previousProjectionMatrix = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        currentProjectionMatrix(3, 4, CV_64F);

    Mat calibrationMatrix(3, 3, CV_64F);
    calibration(calibrationMatrix, CalibrationOption::load);

	while (true) {
//		cap.read(image);
        image = imread("./data/two_frames/1/1.png");
		cvtColor(image, image, COLOR_BGR2GRAY);
		cvtColor(image, image, COLOR_GRAY2BGR);

		// Applied the FAST algorithm to the image and saved the image
		// with the highlighted features in @result
		fastExtractor(image, keypoints, 10);
		drawKeypoints(image, keypoints, result);

		namedWindow("Display Image", WINDOW_AUTOSIZE);
		imshow("Display Image", result);
        waitKey(1000);

        //Transform image into black and white
//		cap.read(image2);
        image2 = imread("./data/two_frames/1/2.png");
		cvtColor(image2, image2, COLOR_BGR2GRAY);
		cvtColor(image2, image2, COLOR_GRAY2BGR);

		std::vector<Point2f> features;
		std::vector<Point2f> newFeatures;
		KeyPoint::convert(keypoints, features);

		trackFeatures(features, image, image2, newFeatures, 10, 10000);

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