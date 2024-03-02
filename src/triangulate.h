#pragma once
#include "opencv2/core/core_c.h"

using namespace cv;

/**
 * This function is a wrapper over reconstructPointsFor3D.
 * First, it converts the transmitted data into computationally convenient data types.
 * Secondly, we allocate an array to store points in 3D.
 * As a result of inner function work, we get three-dimensional points (in homogeneous coordinates).
 *
 * @param projPoints1 and projPoints2 is Nx2 array of feature points in the images.
 *     It can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
 *
 * @param matr1, matr2 3x4 projection matrix of the camera,
 *     i.e. this matrix projects 3D points given in the world's coordinate system into the image.
 *
 * @param points 4D is 4xN array of reconstructed points in homogeneous coordinates.
 *     These points are returned in the world's coordinate system.
 */
void triangulationWrapper(InputArray projPoints1, InputArray projPoints2,
						  const Mat& matr1, const Mat& matr2,
						  OutputArray points4D);

/**
 * Reconstruct 3D points using calibration, rotations, transitions and corresponding points in 2 vectors.
 *
 * @param calibration [in]
 * @param rotation1 [in]
 * @param transition1 [in]
 * @param rotation2 [in]
 * @param transition2 [in]
 * @param points1 [in]
 * @param points2 [in]
 * @param spatialPoints [out] vector with triangulated points
 */
void reconstruct(
		const Mat& calibration,
		const Mat& rotation1, const Mat& transition1,
		const Mat& rotation2, const Mat& transition2,
		const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
		std::vector<Point3f>& spatialPoints
);

/**
 * Normalizes and copies points from homogeneous matrix to vector
 *
 * @param homogeneous3DPoints [in] 4xN matrix with homogeneous points
 * @param spatialPoints [out]
 */
void convertHomogeneousPointsMatrixToSpatialPointsVector(
		const Mat& homogeneous3DPoints, std::vector<Point3f>& spatialPoints
);

/**
 * Converts points from homogeneous format to 3D.
 * @param [in] inputHomogeneous3DPoints input 4xN matrix with homogeneous points
 * @param [out] normalizedHomogeneous3DPoints output 3xN matrix with euclidian 3D points
 */
void normalizeHomogeneousWrapper(const Mat& inputHomogeneous3DPoints, Mat& normalizedHomogeneous3DPoints);

/**
 * Converter to world coordinates.
 * Adds to points current world camera position
 *
 * @param [in,out] points 3xN matrix with points which will be placed in world system
 * @param [in] worldCameraPose 1x3 vector-matrix with world camera position
 * @param [in] worldCameraRotation 3x3 world camera rotation matrix
 */
void placeEuclideanPointsInWorldSystem(Mat& points, Mat& worldCameraPose, Mat& worldCameraRotation);