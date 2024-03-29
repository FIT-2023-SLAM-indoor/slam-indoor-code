#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <iostream>

#include "main_config.h"
#include "cameraCalibration.h"
#include "IOmisc.h"

#define MINIMAL_FOUND_FRAMES_COUNT 10

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
        char c = (char)waitKey(33);
//        Mat resized;
        namedWindow("Test", cv::WINDOW_NORMAL);
        resizeWindow("Test", frame.rows / 32, frame.cols / 32);
        imshow("Test", frame);
//        imwrite("./test.jpg", frame);
        if (c == 27)
            break;
#endif
    }
    if (imagePointsVector.size() < MINIMAL_FOUND_FRAMES_COUNT) {
        std::cerr << "Cannot find enough chessboard frames" << std::endl;
        exit(-1);
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
#ifdef VISUAL_CALIB
            imshow("Test", frame);
            char c = (char)waitKey(500);
            if (c == 27)
                break;
#endif
        }
        else {
            std::cout << "Cannot find chessboard corners" << std::endl;
        }

        if (imagePointsVector.size() >= itersCount)
            break;
    }
    if (imagePointsVector.size() < MINIMAL_FOUND_FRAMES_COUNT) {
        std::cerr << "Cannot detect chessboard on enough count of photos" << std::endl;
        exit(-1);
    }

    Mat cameraMatrixK, distortionCoeffs, R, T;
    calibrateCamera(
            objectPointsVector, imagePointsVector, Size(gray.rows, gray.cols),
            cameraMatrixK, distortionCoeffs, R, T
    );

    if (pathToXML != nullptr)
        saveCalibParametersToXML(pathToXML, cameraMatrixK, distortionCoeffs, R, T);
}