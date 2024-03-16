#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>

#include "config/config.h"
#include "IOmisc.h"

void openLogsStreams() {
	char tmp[256] = "";
	std::string path = configService.getValue<std::string>(ConfigFieldEnum::OUTPUT_DATA_DIR_);
	sprintf(tmp, "%s/main.txt", path.c_str());
	logStreams.mainReportStream.open(tmp);
	sprintf(tmp, "%s/points.txt", path.c_str());
	logStreams.pointsStream.open(tmp);
	sprintf(tmp, "%s/pose.txt", path.c_str());
	logStreams.poseStream.open(tmp);
}

void closeLogsStreams() {
	logStreams.mainReportStream.close();
	logStreams.pointsStream.close();
	logStreams.poseStream.close();
}

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


void rawOutput(const Mat &matrix, std::ofstream &fileStream) {
    // Check if the file stream is opened
    if (!fileStream.is_open()) {
        std::cerr << "Error: stream of file is not opened" << std::endl;
        exit(-1);
    }

    // Write into file every matrix element
    for (int row_id = 0; row_id < matrix.rows; row_id++) {
        for (int col_id = 0; col_id < matrix.cols; col_id++) {
            fileStream << std::fixed << std::setprecision(12) << matrix.at<double>(row_id, col_id);
            // If it wasn't the last element in a current row
            if (col_id < matrix.cols - 1) {
                fileStream << " ";
            }
        }
        fileStream << "\n";
        
        // Check if an error occurred during writing
        if (fileStream.bad()) {
            std::cerr << "Something went wrong during writing in stream" << std::endl;
            exit(-1);
        }
        // Flush the stream to ensure data is written immediately
        fileStream.flush();
    }
}

void rawOutput(const std::vector<Point3f> &vector, std::ofstream &fileStream) {
	Mat pointsMat = Mat(vector);
	pointsMat.reshape(1).convertTo(pointsMat, CV_64F);
	rawOutput(pointsMat, fileStream);
}

void rawOutput(const Mat &matrix, const String &path, const char mode) {
    // Try to open file with received path
    std::ofstream fileStream;
    if (mode == 'a') {
        fileStream.open(path, std::ios_base::app);
    } else if (mode == 'w') {
        fileStream.open(path);
    }
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open or create file with path: " << path << std::endl;
        exit(-1);
    }

    // Write into file every matrix element
    rawOutput(matrix, fileStream);

    fileStream.close();
}
