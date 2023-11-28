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

void calibration(Mat& cameraMatrix, CalibrationOption option, const char *pathToXML) {
    VideoCapture cap;
    std::vector<String> files;
    switch (option) {
        case configureFromWebcam :
            cap.open(0);
            chessboardVideoCalibration(cap);
            break;
        case configureFromVideo :
            cap.open("./data/for_calib.mp4");
            chessboardVideoCalibration(cap);
            break;
        case configureFromFiles :
            glob("./data/for_calib/*.jpg", files, false);
            chessboardPhotosCalibration(files);
            break;
        default:;
    }
    loadMatrixFromXML(pathToXML, cameraMatrix);
}

/**
 * Static function for getting chessboard points.
 *
 * Loads to `objectPoints` points with relative plane sizes (**we assume Z as 0**) of chessboard corners
 * with described shape
 *
 * @param objectPoints Link for loading result
 * @param boardSize Shape of board (`Size(int width, int height)`)
 * @param chessboardSquareSize Rational value of square size. Default `1.0`
 */
static void getObjectPoints(std::vector<Point3f> &objectPoints, Size boardSize, double chessboardSquareSize= 1.0) {
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints.emplace_back(Point3f(
                    j * chessboardSquareSize, i * chessboardSquareSize, 0
            ));
        }
    }
}

void chessboardVideoCalibration(cv::VideoCapture capture, int itersCount, double delay, double squareSize, cv::Size boardSize,
                                const char *pathToXML) {
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

    if (pathToXML != nullptr)
        saveCalibParametersToXML(pathToXML, cameraMatrixK, distortionCoeffs, R, T);
}

void chessboardPhotosCalibration(std::vector<String> &fileNames, int itersCount, double squareSize, Size boardSize,
                                 const char *pathToXML) {
    // Set base plane pattern relative predicted points
    std::vector<Point3f> objectPoints;
    getObjectPoints(objectPoints, boardSize, squareSize);
    std::vector <std::vector<Point3f>> objectPointsVector; // All vector is the same as its first element

    // Recognized points of this pattern
    std::vector<std::vector<Point2f>> imagePointsVector;
    std::vector<Point2f> imagePoints;

    // Recognition
    Mat frame, gray;
    for (const String& filename : fileNames) {
        std::cout << filename << std::endl;
        frame = imread(filename);
        if (frame.empty()) {
            std::cerr << "Empty frame" << std::endl;
            continue;
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // We analyze frame when corners pattern was detected and timedelta >= delay
        if (findChessboardCorners(gray, boardSize, imagePoints)) {
            cornerSubPix(gray, imagePoints, Size(11, 11), Size(-1, -1),
                         { TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,30,0.0001) }
            ); // Make points' corners more precisely
#ifdef VISUAL_CALIB
            drawChessboardCorners(frame, boardSize, imagePoints, true);
#endif
            imagePointsVector.push_back(imagePoints);
            objectPointsVector.push_back(objectPoints);
        }
#ifdef VISUAL_CALIB
        imshow("Test", frame);
        char c = (char)waitKey(500);
        if (c == 27)
            break;
#endif

        if (imagePointsVector.size() >= itersCount)
            break;
    }

    Mat cameraMatrixK, distortionCoeffs, R, T;
    calibrateCamera(
            objectPointsVector, imagePointsVector, Size(gray.rows, gray.cols),
            cameraMatrixK, distortionCoeffs, R, T
    );

    if (pathToXML != nullptr)
        saveCalibParametersToXML(pathToXML, cameraMatrixK, distortionCoeffs, R, T);
}

void saveMatrixToXML(const char *pathToXML, const Mat &matrix, const String& matrixKey, FileStorage::Mode mode) {
    if (mode != FileStorage::WRITE && mode != FileStorage::APPEND)
        std::cerr << "Only WRITE or APPEND mode can be used for save function" << std::endl;
    FileStorage fs;
    if (!fs.open(pathToXML, mode)) {
        std::cerr << format("Cannot open %s", pathToXML) << std::endl;
        exit(-1);
    }
    fs << matrixKey << matrix;
}

void saveCalibParametersToXML(const char *pathToXML, const Mat& cameraMatrixK, const Mat& distortionCoeffs,
                              const Mat& R, const Mat& T) {
    saveMatrixToXML(pathToXML, cameraMatrixK);
    saveMatrixToXML(pathToXML, distortionCoeffs, "DC", FileStorage::APPEND);
    saveMatrixToXML(pathToXML, R, "R", FileStorage::APPEND);
    saveMatrixToXML(pathToXML, T, "T", FileStorage::APPEND);
}


void loadMatrixFromXML(const char *pathToXML, Mat &matrix, const String& matrixKey) {
    FileStorage fs;
    if (!fs.open(pathToXML, FileStorage::READ)) {
        std::cerr << format("Cannot open %s", pathToXML) << std::endl;
        exit(-1);
    }
    fs[matrixKey] >> matrix;
}
