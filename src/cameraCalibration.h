/**
 * Camera calibration header
 */
#include <opencv2/videoio.hpp>
#pragma once

using namespace cv;

/// Calibration wrapper options' enumeration.
enum CalibrationOption {
    configureFromWebcam, ///< Configure using webcam
    configureFromVideo, ///< Configure using video from `"./data/for_calib.mp4"`
    configureFromFiles, ///< Configure using files with pattern `"./data/for_calib/*.jpg"`
    load ///< Just load calibration from file
};

/**
 * Calibration wrapper.
 *
 * Regarding to `option`, loads to `cameraMatrix` camera intrinsic matrix from file specified by `pathToXML`
 * or gets and save this matrix using webcam, video or photos.
 *
 * Paths to video/photos are hardcoded. To specify more calibration parameters use functions below.
 *
 * @param [out] cameraMatrix link to matrix where result will be saved.
 * @param [in] option Option that determines how calibration wrapper will get calibration matrix.
 * @param [in] pathToXML Path to XML file for saving/loading. Relative path starts from project root.
 */
void calibration(Mat& cameraMatrix, CalibrationOption option, const char* pathToXML= "./config/cameraMatrix.xml");

/**
 * Chessboard calibration from video source.
 *
 * Used in `configureFromWebcam` and `configureFromVideo` options.
 *
 * @param [in] capture Video capture. (Can be live stram from camera or video file).
 * @param [in] itersCount Specifies, how many points vectors algorithm need fro calibration.
 * @param [in] delay Specifies delay between successful parsing chessboard corners and next attempt to parse them.
 * @param [in] squareSize Rational size of squares (in mm)
 * @param [in] boardSize Shape of board (`Size(int width, int height)`)
 * @param [in] pathToXML Oath to XML for saving got matrix
 */
void chessboardVideoCalibration(VideoCapture capture, int itersCount= 10, double delay= 3,
                                double squareSize= 23.0, Size boardSize= Size(7, 7),
                                const char* pathToXML= "./config/cameraMatrix.xml");
/**
 * Chessboard calibration from photos.
 *
 * Used in `configureFromFiles` option.
 *
 * @param [in] fileNames Vector of strings with filenames
 * @param [in] itersCount Count of files for calibration
 * @param [in] squareSize Rational size of squares (in mm)
 * @param [in] boardSize Shape of board (`Size(int width, int height)`)
 * @param [in] pathToXML Oath to XML for saving got matrix
 */
void chessboardPhotosCalibration(std::vector<String>& fileNames, int itersCount= 10,
                                double squareSize= 23.0, Size boardSize= Size(7, 7),
                                const char* pathToXML= "./config/cameraMatrix.xml");

/**
 * Function for saving matrix.
 *
 * Saves specified matrix to XML-file with specified tag name.
 *
 * @param [in] pathToXML Path to XML-file.
 * @param [out] matrix Matrix for saving.
 * @param [in] matrixKey Tag name for saving block.
 */
void saveMatrixToXML(const char *pathToXML, const Mat &matrix, const String& matrixKey= "K",
                     FileStorage::Mode mode= FileStorage::WRITE);
/**
 * Save all camera parameters to specified XML.
 *
 * @param [in] pathToXML Path to existing XML-file
 * @param [in] cameraMatrixK 3x3 intrinsic camera
 * @param [in] distortionCoeffs, R, T Other calibration data
 */
void saveCalibParametersToXML(const char *pathToXML, const Mat& cameraMatrixK, const Mat& distortionCoeffs,
                              const Mat& R, const Mat& T);

/**
 * Function for loading matrix.
 *
 * Loads specified by tag name matrix from XML-file.
 *
 * @param [in] pathToXML Path to XML-file.
 * @param [out] matrix Matrix for loading.
 * @param [in] matrixKey Tag name of block with desired matirx.
 */
void loadMatrixFromXML(const char *pathToXML, Mat &matrix, const String& matrixKey= "K");
