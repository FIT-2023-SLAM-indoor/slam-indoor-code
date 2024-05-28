#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "cameraTranslation.h"
#include "../../config/config.h"
#include "../../misc/IOmisc.h"

using namespace cv;

void filterVectorByMask(std::vector<Point2f>& vector, const Mat& mask) {
	Mat filterMask = mask;
	if (filterMask.rows == 1 && filterMask.cols > 1)
		filterMask = filterMask.t();

	int maskSignificantSize = filterMask.rows;
	if (maskSignificantSize != vector.size()) {
		std::cerr << "Incorrect size of filtering mask" << std::endl;
		exit(-1);
	}

	std::vector<Point2f> newVector;
	for (int i = 0; i < filterMask.rows; i++) {
        if (filterMask.at<uchar>(i)) {
            newVector.push_back(vector[i]);
		}
    }
	vector = newVector;
}

bool estimateTransformation(
		const std::vector<Point2f>& points1, const std::vector<Point2f>& points2, const Mat& calibrationMatrix,
		Mat& rotationMatrix, Mat& translationVector, Mat& chiralityMask
) {
	Mat mask;
	double maskNonZeroElemsCnt = 0;
	bool useRANSAC = configService.getValue<bool>(ConfigFieldEnum::RP_USE_RANSAC);
	Mat essentialMatrix;
	if (useRANSAC)
		essentialMatrix = findEssentialMat(
			points1, points2, calibrationMatrix, RANSAC,
		   configService.getValue<double>(ConfigFieldEnum::RP_RANSAC_PROB),
		   configService.getValue<double>(ConfigFieldEnum::RP_RANSAC_THRESHOLD),
		   mask
	   );
	else
		essentialMatrix = findEssentialMat(points1, points2, calibrationMatrix);
	if (essentialMatrix.empty())
		return false;

	if (useRANSAC) {
		maskNonZeroElemsCnt = countNonZero(mask);
		logStreams.mainReportStream << "Points' count used in RANSAC E matrix estimation: " << maskNonZeroElemsCnt << std::endl;
//    if ((maskNonZeroElemsCnt / points1.size()) < RANSAC_GOOD_POINTS_PERCENT)
//        return false;
	}

	// Find P matrix using wrapped OpenCV SVD and triangulation
	int passedPointsCount = recoverPose(
		essentialMatrix, points1, points2, calibrationMatrix,
		rotationMatrix, translationVector,
		configService.getValue<double>(ConfigFieldEnum::RP_DISTANCE_THRESHOLD),
		chiralityMask
	);
	maskNonZeroElemsCnt = countNonZero(chiralityMask);
	logStreams.mainReportStream << "Points passed chirality check count: " << maskNonZeroElemsCnt << std::endl;
	return passedPointsCount > 0;
}

void refineTransformationForGlobalCoords(
	Mat& worldCameraRotation, Mat& worldCameraPose,
	Mat& rotationMatrix, Mat& translationVector
) {
	translationVector = worldCameraPose + rotationMatrix * translationVector;
	rotationMatrix = worldCameraRotation * rotationMatrix;
}

void addHomogeneousRow(Mat& m) {
    Mat row = Mat::zeros(1, m.cols, CV_64F);
    row.at<double>(0, m.cols-1) = 1;
    m.push_back(row);
}

void removeHomogeneousRow(Mat& m) {
    m.pop_back();
}
