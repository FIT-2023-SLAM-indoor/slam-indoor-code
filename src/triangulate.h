#pragma once
#include "opencv2/core/core_c.h"

/**
 * This function is a wrapper over reconstructPointsFor3D.
 * First, it converts the transmitted data into computationally convenient data types.
 * Secondly, we allocate an array to store points in 3D.
 * As a result of inner function work, we get three-dimensional points (in homogeneous coordinates).
 *
 * @param projPoints1 and projPoints2 is Nx2 array of feature points in the images.
 *     It can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
 *
 * @param matr1 and matr2 3x4 projection matrix of the camera,
 *     i.e. this matrix projects 3D points given in the world's coordinate system into the image.
 *
 * @param points 4D is 4xN array of reconstructed points in homogeneous coordinates.
 *     These points are returned in the world's coordinate system.
 */
void triangulate(cv::InputArray projPoints1, cv::InputArray projPoints2,
    cv::Mat matr1, cv::Mat matr2,
    cv::OutputArray points4D);