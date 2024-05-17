#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>

#include "config/config.h"
#include "./cycleProcessing/mainCycleStructures.h"

#include "IOmisc.h"

void openLogsStreams(LogFilesStreams &streams, std::_Ios_Openmode mode) {
	char tmp[256] = "";
	std::string path = configService.getValue<std::string>(ConfigFieldEnum::OUTPUT_DATA_DIR);
	sprintf(tmp, "%s/main.txt", path.c_str());
	streams.mainReportStream.open(tmp, mode);
	sprintf(tmp, "%s/points.txt", path.c_str());
	streams.pointStream.open(tmp, mode);
	sprintf(tmp, "%s/colors.txt", path.c_str());
	streams.colorStream.open(tmp, mode);
	sprintf(tmp, "%s/poses.txt", path.c_str());
	streams.poseStream.open(tmp, mode);
	sprintf(tmp, "%s/rotations.txt", path.c_str());
	streams.rotationStream.open(tmp, mode);
	sprintf(tmp, "%s/time.txt", path.c_str());
	streams.timeStream.open(tmp, mode);
}

void closeLogsStreams(LogFilesStreams &streams) {
	streams.mainReportStream.close();
	streams.pointStream.close();
	streams.colorStream.close();
	streams.poseStream.close();
	streams.rotationStream.close();
	streams.timeStream.close();
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

void rawOutput(const Mat &matrix, std::fstream &fileStream) {
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

void rawOutput(const SpatialPointsVector &vector, std::fstream &fileStream) {
	Mat pointsMat = Mat(vector);
	pointsMat.reshape(1).convertTo(pointsMat, CV_64F);
	rawOutput(pointsMat, fileStream);
}

void rawOutput(const std::vector<Vec3b> &vector, std::fstream &fileStream) {
	Mat colorsMat = Mat(vector);
	colorsMat.reshape(1).convertTo(colorsMat, CV_64F);
	rawOutput(colorsMat, fileStream);
}

void rawOutput(const std::vector<Point3f> &vector, std::fstream &fileStream) {
	Mat pointsMat = Mat(vector);
	pointsMat.reshape(1).convertTo(pointsMat, CV_64F);
	rawOutput(pointsMat, fileStream);
}

void printDivider(std::fstream &stream) {
	stream << std::endl << "================================================================\n" << std::endl;
}

GlobalData getGlobalDataFromLogFiles() {
	GlobalData globalData;
	LogFilesStreams inputLogStreams;
	openLogsStreams(inputLogStreams, std::ios_base::in);
	while (!inputLogStreams.poseStream.eof()) {
		double x, y, z;
		inputLogStreams.poseStream >> x >> y >> z;
		if (inputLogStreams.poseStream.eof())
			break;
		Mat pose = (Mat_<double>(1, 3) << x, y, z);
		globalData.spatialCameraPositions.push_back(pose.clone().t());

		double rData[9] = {};
		for (int i = 0; i < 9; ++i)
			inputLogStreams.rotationStream >> rData[i];
		Mat rotation(3, 3, CV_64F, rData);
		globalData.cameraRotations.push_back(rotation.clone());
	}
	if (globalData.cameraRotations.size() != globalData.spatialCameraPositions.size()) {
		std::cerr << "Count of rotations and translations must be equal" << std::endl;
		std::cerr << "Actual rotations count: " << globalData.cameraRotations.size() << std::endl;
		std::cerr << "Actual translations count: " << globalData.spatialCameraPositions.size() << std::endl;
		throw std::exception();
	}
	while (!inputLogStreams.pointStream.eof()) {
		Point3d p;
		inputLogStreams.pointStream >> p.x >> p.y >> p.z;
		if (inputLogStreams.pointStream.eof())
			break;
		globalData.spatialPoints.push_back(p);

		double r, g, b;
		inputLogStreams.colorStream >> r >> g >> b;
		Vec3b color((uchar) r, (uchar) g, (uchar) b);
		globalData.spatialPointsColors.push_back(color);
	}
	if (globalData.spatialPoints.size() != globalData.spatialPointsColors.size()) {
		std::cerr << "Count of points and their colors must be equal" << std::endl;
		std::cerr << "Actual points count: " << globalData.spatialPoints.size() << std::endl;
		std::cerr << "Actual colors count: " << globalData.spatialPointsColors.size() << std::endl;
		throw std::exception();
	}
	closeLogsStreams(inputLogStreams);

	return globalData;
}