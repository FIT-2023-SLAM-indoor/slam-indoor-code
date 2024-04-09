#pragma once

#include <opencv2/videoio.hpp>
#include "fstream"

#include "./cycleProcessing/mainCycleStructures.h"

using namespace cv;

typedef struct LogFilesStreams {
	std::ofstream mainReportStream;
	std::ofstream pointsStream;
	std::ofstream poseStream;
	std::ofstream extractedMatchedTable;
} LogFilesStreams;

extern LogFilesStreams logStreams;

/**
 * Opens streams for fields of LogFilesStreams' global structure.
 */
void openLogsStreams();

/**
 * Closes streams for fields of LogFilesStreams' global structure.
 */
void closeLogsStreams();

/**
 * Sorts files photos paths by its numbers.
 *
 * @param paths string names' vector
 */
void sortGlobs(std::vector<String>& paths);

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
                     FileStorage::Mode mode=FileStorage::WRITE);
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
void loadMatrixFromXML(const char *pathToXML, Mat &matrix, const String& matrixKey="K");

/**
 * Writes the elements of a given matrix to an output file stream.
 *
 * @param [in] matrix The input matrix.
 * @param [in] fileStream The output file stream to write the matrix elements to.
 *
 * @throws std::runtime_error if the file stream is not opened or if an error occurs during writing.
 */
void rawOutput(const Mat &matrix, std::ofstream &fileStream);

/**
 * Works as function above but at first converts vector to matrix.
 *
 * @param [in] vector
 * @param [out] fileStream
 */
void rawOutput(const SpatialPointsVector &vector, std::ofstream &fileStream);

/**
 * Works as function above but at first converts vector to matrix.
 *
 * @param [in] vector
 * @param [out] fileStream
 */
void rawOutput(const std::vector<Point3f> &vector, std::ofstream &fileStream);

/**
 * Writes the elements of a given matrix to a file at the specified path.
 *
 * @param [in] matrix The input matrix.
 * @param [in] path The path to the file where the matrix will be written.
 * @param [in] mode The mode of writing to a file. 'w' - overwriting the file, 'a' - writing to the end.
 *
 * @throws std::runtime_error if the file cannot be opened or created.
 */
void rawOutput(const Mat &matrix, const String &path, const char mode='a');
