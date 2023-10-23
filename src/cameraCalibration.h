#include <opencv2/videoio.hpp>
#pragma once

using namespace cv;

enum CalibrationOption { configureFromWebcam, configureFromVideo, configureFromFiles, load };

void calibration(Mat& cameraMatrix, CalibrationOption option,
                 const char* pathToXML= "./config/cameraMatrix.xml", int test= 0);

static void getObjectPoints(std::vector<Point3f>& objectPoints, Size boardSize, double chessboardSquareSize=1.0);

void chessboardVideoCalibration(VideoCapture capture, int itersCount= 10, double delay= 3,
                                double squareSize= 23.0, Size boardSize= Size(7, 7),
                                const char* pathToXML= "./config/cameraMatrix.xml");

void chessboardPhotosCalibration(std::vector<String>& fileNames, int itersCount= 10,
                                double squareSize= 23.0, Size boardSize= Size(7, 7),
                                const char* pathToXML= "./config/cameraMatrix.xml");

void saveCalibration(const char *pathToXML, Mat &cameraMatrix);

void loadCalibration(const char *pathToXML, Mat &cameraMatrix);
