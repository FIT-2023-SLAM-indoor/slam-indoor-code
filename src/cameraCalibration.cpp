#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <iostream>

#include "cameraCalibration.h"

#define VISUAL_CALIB

using namespace cv;

/*
 * Calculates real relative point of real world plane pattern
 */
void getObjectPoints(std::vector<Point3f> &objectPoints, Size boardSize, double chessboardSquareSize) {
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints.emplace_back(Point3f(
                    j * chessboardSquareSize, i * chessboardSquareSize, 0
            ));
        }
    }
}

void chessboardCalibration(cv::VideoCapture capture, int itersCount, double delay, double squareSize, cv::Size boardSize,
                           const char *pathToSaveFile) {
    // Set base plane pattern relative predicted points
    std::vector<Point3f> objectPoints;
    getObjectPoints(objectPoints, boardSize, squareSize);
    std::vector <std::vector<Point3f>> objectPointsVector; // All vector is the same as its first element

    // Recognized points of this pattern
    std::vector<std::vector<Point2f>> imagePointsVector;
    std::vector<Point2f> imagePoints;

    // Recognition
    Mat frame, gray;
    clock_t prevClock = 0;
    while (imagePointsVector.size() < itersCount) {
        capture.read(frame);
        if (frame.empty()) {
            std::cerr << "Empty frame" << std::endl;
            exit(-1);
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // We analyze frame when corners pattern was detected and timedelta >= delay
        if (findChessboardCorners(gray, boardSize, imagePoints) &&
            clock() - prevClock > delay*CLOCKS_PER_SEC) {
            cornerSubPix(gray, imagePoints, Size(11, 11), Size(-1, -1),
                         { TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,30,0.0001) }
            ); // Make points' corners more precisely
#ifdef VISUAL_CALIB
            drawChessboardCorners(frame, boardSize, imagePoints, true);
#endif
            imagePointsVector.push_back(imagePoints);
            objectPointsVector.push_back(objectPoints);
            prevClock = clock();
        }
#ifdef VISUAL_CALIB
        imshow("Test", frame);
        char c = (char)waitKey(33);
        if (c == 27)
            break;
#endif
    }

    Mat cameraMatrixK, distortionCoeffs, R, T;
    calibrateCamera(
            objectPointsVector, imagePointsVector, Size(gray.rows, gray.cols),
            cameraMatrixK, distortionCoeffs, R, T
    );

    FileStorage fs;
    if (!fs.open(pathToSaveFile, FileStorage::WRITE)) {
        std::cerr << format("Cannot open %s", pathToSaveFile) << std::endl;
        exit(-1);
    }
    fs << "K" << cameraMatrixK;
}


void loadCalibration(const char *pathToXML, Mat &cameraMatrix) {
    FileStorage fs;
    if (!fs.open(pathToXML, FileStorage::READ)) {
        std::cerr << format("Cannot open %s", pathToXML) << std::endl;
        exit(-1);
    }
    fs["K"] >> cameraMatrix;
}
