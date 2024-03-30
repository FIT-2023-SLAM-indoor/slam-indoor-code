#pragma once

#include "opencv2/core.hpp"
#include "ceres/ceres.h"
#include "mainCycle.h"

/**
 * Functor class for re-projection cost function `project(K*[R|T]*P3d) - p2d`
 */
class ProjectionCostFunctor {
public:
    cv::Point2d imagePoint;

    /**
     * Base constructor.
     *
     * @param imagePoint known 2d point position.
     */
    ProjectionCostFunctor(cv::Point2d imagePoint);

    /**
     * Operator with main part of cost function
     *
     * @tparam T parameters' type (it's likely to be double)
     * @param calibration array with contains focal lengths and optical centers of camera
     * @param rotation array with 3 rotations axis computed using Rodrigues
     * @param transition transition vector
     * @param point3d 3D point for re-projection
     * @param residuals result values (actually [0] is a X axis subtraction and [1] is an Y axis subtraction)
     * @return always true
     */
    template<typename T>
    bool operator()(
            const T* const calibration, const T* const rotation, const T* const transition,
            const T* const point3d, T *residuals
    ) const;

    static ceres::CostFunction* createFunctor(cv::Point2d imagePoint);
};

void bundleAdjustment(
        cv::Mat& calibrationMatrix,
		std::vector<TemporalImageData> &imagesDataForAdjustment,
		GlobalData &globalData
);
