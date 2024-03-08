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
 * @param vector [in,out] vector which will be filtered
 * @param mask [in] mask for filtration
 */
void filterVectorByMask(std::vector<Point2f>& vector, const Mat& mask);

// TODO: стоит убрать после полной переработки
/**
 * OUTDATED.
 * 
 * Function for estimation projection matrix and its parts according to calibration data and two 2D points vectors.
 *
 * @param [in] points1 the first points' vector
 * @param [in] points2 the first points' vector
 * @param [in] calibrationMatrix 3x3. see `cameraCalibration.h` for methods of getting this matrix
 * @param [out] rotationMatrix 3x3 rotation between every two points
 * @param [out] translationVector 3x1 translation between every two points
 * @param [out] projectionMatrix 3x4 projection matrix composed from rotation and translation as [R|t]
 * @return true weather new projection matrix was estimated correctly (one of four possible P matrices project points
 *     with positive Z
 */
bool estimateProjection(
	std::vector<Point2f>& points1, std::vector<Point2f>& points2, const Mat& calibrationMatrix,
	Mat& rotationMatrix, Mat& translationVector, Mat& projectionMatrix, Mat& triangulatedPoints
);

/**
 * Refiner for world point nad world rotation.
 * In common case, updates any 3D point by rotation and translation
 *
 * @param [out] rotationMatrix 3x3 rotation matrix
 * @param [out] translationVector 3x1 translation vector-matrix
 * @param [in,out] worldCameraPose 1x3 vector-matrix with camera world pose which will be refined by estimated rotation and
 *     translation
 * @param [in,out] worldCameraRotation 3x3 global rotation matrix
 */
void refineWorldCameraPose(
	Mat& rotationMatrix, Mat& translationVector,
	Mat& worldCameraPose, Mat& worldCameraRotation
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
