#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <iostream>
#include <fstream>

#include "main_config.h"
#include "IOmisc.h"


void sortGlobs(std::vector<String> &paths) {
    std::sort(paths.begin(), paths.end(), [](const String &a, const String &b) {
        int aLen = a.length(), bLen = b.length();
        if (aLen != bLen)
            return aLen < bLen;
        int startDigitIndex = 5;
        int aNum = 0, bNum = 0, p = 1;
        while (isdigit(a[aLen - 1 - startDigitIndex])) {
            aNum += (a[aLen - 1 - startDigitIndex] - '0') * p;
            bNum += (b[aLen - 1 - startDigitIndex] - '0') * p;
            p *= 10;
            startDigitIndex++;
        }
        return a < b; 
    });
}

void saveMatrixToXML(
    const char *pathToXML, const Mat &matrix, const String &matrixKey, FileStorage::Mode mode)
{

    if (mode != FileStorage::WRITE && mode != FileStorage::APPEND)
        std::cerr << "Only WRITE or APPEND mode can be used for save function" << std::endl;
    FileStorage fs;
    if (!fs.open(pathToXML, mode))
    {
        std::cerr << format("Cannot open %s", pathToXML) << std::endl;
        exit(-1);
    }
    fs << matrixKey << matrix;
}

void saveCalibParametersToXML(
    const char *pathToXML, const Mat &cameraMatrixK, 
    const Mat &distortionCoeffs, const Mat &R, const Mat &T)
{
    saveMatrixToXML(pathToXML, cameraMatrixK);
    saveMatrixToXML(pathToXML, distortionCoeffs, "DC", FileStorage::APPEND);
    saveMatrixToXML(pathToXML, R, "R", FileStorage::APPEND);
    saveMatrixToXML(pathToXML, T, "T", FileStorage::APPEND);
}

void loadMatrixFromXML(const char *pathToXML, Mat &matrix, const String &matrixKey) {
    FileStorage fs;
    if (!fs.open(pathToXML, FileStorage::READ))
    {
        std::cerr << format("Cannot open %s", pathToXML) << std::endl;
        exit(-1);
    }
    fs[matrixKey] >> matrix;
}


void rawOutput(const Mat &matrix, const String &path) {
    // Try to open file with received path
    std::ofstream file_stream(path);
    if (!file_stream.is_open()) {
        std::cerr << "Failed to open or create file with path: " << path << std::endl;
        exit(-1);
    }

    // Write into file every matrix element
    for (int row_id = 0; row_id < matrix.rows; row_id++) {
        for (int col_id = 0; col_id < matrix.cols; col_id++) {
            file_stream << matrix.at<double>(row_id, col_id);
            // If it wasn't the last element in a current row
            if (col_id < matrix.cols - 1) {
                file_stream << " ";
            }
        }
        file_stream << "\n";
    }
    
    file_stream.close();
}