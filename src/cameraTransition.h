#pragma once
#include <opencv2/calib3d.hpp>

using namespace cv;

/**
 * Function for estimation rotation and transition between 2 2D points' vectors.
 *
 * @param points1 [in]
 * @param points2 [in]
 * @param calibrationMatrix [in]
 * @param rotationMatrix [out]
 * @param translationVector [out]
 * @param mask [out] matrix of size Nx1 where value [i][0] marks that point has passed chirality check and can be used for triangulation
 * @return true when rotation and transition was estimated successfully and likely correct
 */
bool estimateTransformation(
	const std::vector<Point2f>& points1, const std::vector<Point2f>& points2, const Mat& calibrationMatrix,
	Mat& rotationMatrix, Mat& translationVector, Mat& chiralityMask
);

/**
 * Filter-function for vector by specified mask.
 *
 * For vector of size N mask can be either 1xN either Nx1
 *
 * @param [in,out] vector vector which will be filtered
 * @param [in] mask mask for filtration
 */
void filterVectorByMask(std::vector<Point2f>& vector, const Mat& mask);


/**
 *
 * @param [in] worldCameraRotation 3x3 rotation matrix
 * @param [out] worldCameraPose 3x1 translation vector-matrix
 * @param [in,out] rotationMatrix 3x3 rotation matrix which will be refine by local rotation
 * @param [in,out] translationVector 3x1 translation vector-matrix which will be refined by estimated
 * 	   combination of local rotation and translation
 */
void refineTransformationForGlobalCoords(
	Mat& worldCameraRotation, Mat& worldCameraPose,
	Mat& rotationMatrix, Mat& translationVector
);

/**
 * Transformer for adding "homogeneous" row.
 * @param [in,out] m matrix for where will be added row contains zeros and 1 on the last position
 */
void addHomogeneousRow(Mat& m);

/**
 * Removes last row in matrix
 * @param [in, out] m matrix where will be removed last row
 */
void removeHomogeneousRow(Mat& m);
