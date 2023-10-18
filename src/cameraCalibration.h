#include <opencv2/videoio.hpp>
#pragma once

using namespace cv;

static void getObjectPoints(std::vector<Point3f>& objectPoints, Size boardSize, double chessboardSquareSize=1.0);

void chessboardCalibration(VideoCapture capture, int itersCount=10, double delay=3,
                           double squareSize=23.0, Size boardSize=Size(7, 7),
                           const char* pathToSaveFile="./config/cameraMatrix.xml");

void saveCalibration();

void loadCalibration(const char *pathToXML, Mat &cameraMatrix);
