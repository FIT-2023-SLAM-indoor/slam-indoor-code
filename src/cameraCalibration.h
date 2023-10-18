#include <opencv2/videoio.hpp>
#pragma once;

static void getObjectPoints(std::vector<cv::Vec3f>& objectPoints, cv::Size boardSize, double chessboardSquareSize);

void calibrate(cv::VideoCapture capture);

void saveCalibration();

void loadCalibration();