#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <iostream>

#include "cameraCalibration.h"

using namespace cv;

static void getObjectPoints(std::vector<Point3f>& objectPoints, Size boardSize, double chessboardSquareSize) {
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints.push_back(Point3f(
                j * chessboardSquareSize, i * chessboardSquareSize, 0
            ));
        }
    }
}

void calibrate(VideoCapture capture) {
    Size boardSize(7, 7);
    Mat frame, gray;
    std::vector<Point3f> objectPoints;
    getObjectPoints(objectPoints, boardSize, 1);
    std::vector <std::vector<Point3f>> allObjectPoints;
    std::vector<std::vector<Point2f>> savedImagePoints;
    std::vector<Point2f> corners; 
    clock_t prevClock = 0;
    while (savedImagePoints.size() < 10) {
        capture.read(frame);
        if (frame.empty()) {
            std::cerr << "Empty frame" << std::endl;
            exit(-1);
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if (findChessboardCorners(gray, boardSize, corners) && 
            clock() - prevClock > 5*CLOCKS_PER_SEC) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                { TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,30,0.0001) }
            );
            drawChessboardCorners(frame, boardSize, corners, true);
            savedImagePoints.push_back(corners);
            allObjectPoints.push_back(objectPoints);
            prevClock = clock();
        }
        imshow("Test", frame);
        char c = (char)waitKey(33);
        if (c == 27)
            break;
    }
    
    Mat cameraMatrixK, distortionCoeffs, R, T;
    calibrateCamera(
        allObjectPoints, savedImagePoints, Size(gray.rows, gray.cols),
        cameraMatrixK, distortionCoeffs, R, T
    );

    std::cout << cameraMatrixK << std::endl;
}