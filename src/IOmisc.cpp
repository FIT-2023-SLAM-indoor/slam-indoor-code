#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>

#include "config/config.h"
#include "./cycleProcessing/mainCycleStructures.h"

#include "IOmisc.h"

void openLogsStreams() {
	char tmp[256] = "";
	std::string path = configService.getValue<std::string>(ConfigFieldEnum::OUTPUT_DATA_DIR);
	sprintf(tmp, "%s/main.txt", path.c_str());
	logStreams.mainReportStream.open(tmp);
	sprintf(tmp, "%s/points.txt", path.c_str());
	logStreams.pointStream.open(tmp);
	sprintf(tmp, "%s/colors.txt", path.c_str());
	logStreams.colorStream.open(tmp);
	sprintf(tmp, "%s/poses.txt", path.c_str());
	logStreams.poseStream.open(tmp);
	sprintf(tmp, "%s/rotations.txt", path.c_str());
	logStreams.rotationStream.open(tmp);
	sprintf(tmp, "%s/time.txt", path.c_str());
	logStreams.timeStream.open(tmp);
}

void closeLogsStreams() {
	logStreams.mainReportStream.close();
	logStreams.pointStream.close();
	logStreams.colorStream.close();
	logStreams.poseStream.close();
	logStreams.rotationStream.close();
	logStreams.timeStream.close();
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
    if (!fileStream.is_open()) {
        std::cerr << "Error: stream of file is not opened" << std::endl;
        exit(-1);
    }

    for (int row_id = 0; row_id < matrix.rows; row_id++) {
        for (int col_id = 0; col_id < matrix.cols; col_id++) {
            fileStream << std::fixed << std::setprecision(12) << matrix.at<double>(row_id, col_id);
            if (col_id < matrix.cols - 1) {
                fileStream << " ";
            }
        }
        fileStream << "\n";
        
        if (fileStream.bad()) {
            std::cerr << "Something went wrong during writing in stream" << std::endl;
            exit(-1);
        }
        fileStream.flush();
    }
}

void rawOutput(const SpatialPointsVector &vector, std::ofstream &fileStream) {
	Mat pointsMat = Mat(vector);
	pointsMat.reshape(1).convertTo(pointsMat, CV_64F);
	rawOutput(pointsMat, fileStream);
}

void rawOutput(const std::vector<Vec3b> &vector, std::ofstream &fileStream) {
	Mat colorsMat = Mat(vector);
	colorsMat.reshape(1).convertTo(colorsMat, CV_64F);
	rawOutput(colorsMat, fileStream);
}

void rawOutput(const std::vector<Point3f> &vector, std::ofstream &fileStream) {
	Mat pointsMat = Mat(vector);
	pointsMat.reshape(1).convertTo(pointsMat, CV_64F);
	rawOutput(pointsMat, fileStream);
}

void printDivider(std::ofstream &stream) {
	stream << std::endl << "================================================================\n" << std::endl;
}