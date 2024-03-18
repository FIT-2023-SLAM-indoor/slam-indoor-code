#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"

#include "config/config.h"
#include "bundleAdjustment.h"

ProjectionCostFunctor::ProjectionCostFunctor(cv::Point2d imagePoint) : imagePoint(imagePoint) {}

template <typename T>
bool ProjectionCostFunctor::operator()(
        const T* const calibration, const T* const rotation, const T* const transition,
        const T* const point3d, T *residuals
) const {
    T evaluated3dPoint[3];
    ceres::AngleAxisRotatePoint(rotation, point3d, evaluated3dPoint);
    for (int i = 0; i < 3; ++i)
        evaluated3dPoint[i] += transition[i];

    T x2d = evaluated3dPoint[0] / evaluated3dPoint[2];
    T y2d = evaluated3dPoint[1] / evaluated3dPoint[2];

    T fx = calibration[0];
    T fy = calibration[1];
    T cx = calibration[2];
    T cy = calibration[3];

    T predictedX = fx * x2d + cx;
    T predictedY = fy * y2d + cy;

    residuals[0] = predictedX - this->imagePoint.x;
    residuals[1] = predictedY - this->imagePoint.y;

    return true;
}

ceres::CostFunction* ProjectionCostFunctor::createFunctor(cv::Point2d imagePoint) {
    return new ceres::AutoDiffCostFunction<ProjectionCostFunctor, 2, 4, 3, 3, 3>( new ProjectionCostFunctor(imagePoint) );
}

void bundleAdjustment(
        cv::Mat& calibrationMatrix,
        std::vector<cv::Mat*>& projectionMatrixVector,
        std::vector<cv::Mat*>& points3dVector,
        std::vector<cv::Mat*>& points2dVector
) {
    ceres::Problem problem; // Pogovorim ob etom?

    double calibrationArray[] = {
        calibrationMatrix.at<double>(0, 0), calibrationMatrix.at<double>(1, 1),
        calibrationMatrix.at<double>(0, 2), calibrationMatrix.at<double>(1, 2)
    };
    // Here can be used addParameterBlockMethods but I don't understand reason to use them

    for (int i = 0; i < projectionMatrixVector.size(); ++i) {
        cv::Mat r;
        cv::Rodrigues(projectionMatrixVector.at(i)->colRange(0, 3), r);
        double *rotation = r.ptr<double>();
        double *transition = projectionMatrixVector.at(i)->col(3).ptr<double>();

        ceres::LossFunction *lossFunction = new ceres::HuberLoss(
			configService.getValue<double>(ConfigFieldEnum::BA_HUBER_LOSS_FUNCTION_PARAMETER)
		);

        cv::Mat *points3d = points3dVector.at(i);
        cv::Mat *points2d = points2dVector.at(i);
        for (int j = 0; j < points3d->rows; ++j) {
            cv::Point2d point2d(points2d->at<double>(j, 0), points2d->at<double>(j, 1));
            ceres::CostFunction *costFunction = ProjectionCostFunctor::createFunctor(point2d);
            std::vector<const double*> parameters;
            problem.AddResidualBlock(
				costFunction, lossFunction,
				calibrationArray, rotation, transition, points3d->row(j).ptr<double>()
            );
        }
    }

    ceres::Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = configService.getValue<int>(ConfigFieldEnum::BA_THREADS_CNT);
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        std::cout << "BA failed" << std::endl;
        return;
    }
    std::cout << "Bundle Adjustment statistics (approximated RMSE):" << std::endl
              << " #residuals: " << summary.num_residuals << std::endl
              << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << std::endl
              << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << std::endl
              << " Time (s): " << summary.total_time_in_seconds << std::endl;
}