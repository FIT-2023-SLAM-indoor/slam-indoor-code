#pragma once

#include "main_config.h"
#include <opencv2/videoio.hpp>

using namespace cv;

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
void loadMatrixFromXML(const char *pathToXML, Mat &matrix, const String& matrixKey="K");
