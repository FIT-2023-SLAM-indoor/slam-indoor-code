#pragma once
#include <opencv2/calib3d.hpp>

/**
 * Function for estimation projection matrix and its parts according to calibration data and two 2D points arrays which
 * are matrices with size Nx2.
 *
 * @param [in] points1 the first points' matrix
 * @param [in] points2 the first points' matrix
 * @param [in] calibrationMatrix 3x3. see `cameraCalibration.h` for methods of getting this matrix
 * @param [out] rotationMatrix 3x3 rotation between every two points
 * @param [out] translationVector 3x1 translation between every two points
 * @param [out] projectionMatrix 3x4 projection matrix composed from rotation and translation as [R|t]
 * @return true weather new projection matrix was estimated correctly (one of four possible P matrices project points with positive Z
 */
bool estimateProjection(cv::InputArray points1, cv::InputArray points2, const cv::Mat& calibrationMatrix,
                        cv::Mat& rotationMatrix, cv::Mat& translationVector, cv::Mat& projectionMatrix);
/**
 * Main cycle test with frames-pairs.
 *
 * @param [in] FEATURE_EXTRACTING_THRESHOLD
 * @param [in] FEATURE_TRACKING_BARRIER
 * @param [in] FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE
 */
void reportingCycleForFramesPairs(int FEATURE_EXTRACTING_THRESHOLD, int FEATURE_TRACKING_BARRIER,
                                  int FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE);