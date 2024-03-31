#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"

#include "../config/config.h"
#include "../cycle_processing/mainCycleStructures.h"

#include "bundleAdjustment.h"

ProjectionCostFunctor::ProjectionCostFunctor(cv::Point2d &imagePoint) : imagePoint(imagePoint) {}

template <typename T>
bool ProjectionCostFunctor::operator()(
	const T* const calibration, const T* const extrinisics,
	const T* const point3d, T *residuals
) const {
    T evaluated3dPoint[3];
	ceres::AngleAxisRotatePoint(extrinisics, point3d, evaluated3dPoint);
	evaluated3dPoint[0] += extrinisics[3];
	evaluated3dPoint[1] += extrinisics[4];
	evaluated3dPoint[2] += extrinisics[5];

    T x2d = evaluated3dPoint[0] / evaluated3dPoint[2];
    T y2d = evaluated3dPoint[1] / evaluated3dPoint[2];

    T fx = calibration[0];
    T fy = calibration[1];
    T cx = calibration[2];
    T cy = calibration[3];

    T predictedX = fx * x2d + cx;
    T predictedY = fy * y2d + cy;

    residuals[0] = predictedX - T(imagePoint.x);
    residuals[1] = predictedY - T(imagePoint.y);

    return true;
}

ceres::CostFunction* ProjectionCostFunctor::createFunctor(cv::Point2d imagePoint) {
    return new ceres::AutoDiffCostFunction<ProjectionCostFunctor, 2, 4, 6, 3>( new ProjectionCostFunctor(imagePoint) );
}

/**
 *
 * @param [in] calibrationMatrix
 * @param [in] imagesDataForAdjustment
 * @param [out] calibration
 * @param [out] extrinsicsVector
 */
static void convertDataForBA(
	cv::Mat& calibrationMatrix, std::vector<TemporalImageData> &imagesDataForAdjustment,
	double* &calibration, std::vector<double*> &extrinsicsVector
);

/**
 *
 * @param [in] calibration
 * @param [in] extrinsicsVector
 * @param [out] calibrationMatrix
 * @param [out] imagesDataForAdjustment
 */
static void convertDataFromBA(
	double* calibration, std::vector<double*> &extrinsicsVector,
	cv::Mat& calibrationMatrix, std::vector<TemporalImageData> &imagesDataForAdjustment
);

void bundleAdjustment(
	cv::Mat& calibrationMatrix,
	std::vector<TemporalImageData> &imagesDataForAdjustment,
	GlobalData &globalData
) {
    ceres::Problem problem; // Pogovorim ob etom?

	double* calibration;
	std::vector<double*> extrinsicsVector;
	convertDataForBA(calibrationMatrix, imagesDataForAdjustment, calibration, extrinsicsVector);

	for (auto extrinsics : extrinsicsVector)
		problem.AddParameterBlock(extrinsics, 6);
	problem.SetParameterBlockConstant(extrinsicsVector[0]);

	ceres::LossFunction *lossFunction = new ceres::HuberLoss(
		configService.getValue<double>(ConfigFieldEnum::BA_HUBER_LOSS_FUNCTION_PARAMETER)
	);
    for (int i = 0; i < extrinsicsVector.size(); ++i) {
		auto &imageData = imagesDataForAdjustment.at(i);
		std::vector<int> corresponds = imageData.correspondSpatialPointIdx;
		std::vector<cv::KeyPoint> points = imageData.allExtractedFeatures;
        for (int pointIdx = 0; pointIdx < points.size(); ++pointIdx) {
			int correspondIdx = corresponds.at(pointIdx);
			if (correspondIdx < 0)
				continue;

            cv::Point2d point2d = points[pointIdx].pt;
            ceres::CostFunction *costFunction = ProjectionCostFunctor::createFunctor(point2d);
            problem.AddResidualBlock(
				costFunction, nullptr,
				calibration, extrinsicsVector.at(i),
				&(globalData.spatialPoints[correspondIdx].x)
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

	if (!summary.IsSolutionUsable())
		std::cout << "BA failed" << std::endl;
	else
	    std::cout << "Bundle Adjustment statistics (approximated RMSE):" << std::endl
				  << " #residuals: " << summary.num_residuals << std::endl
				  << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << std::endl
				  << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << std::endl
				  << " Time (s): " << summary.total_time_in_seconds << std::endl;
	convertDataFromBA(calibration, extrinsicsVector, calibrationMatrix, imagesDataForAdjustment);
}

static void convertDataForBA(
	cv::Mat& calibrationMatrix, std::vector<TemporalImageData> &imagesDataForAdjustment,
	double* &calibration, std::vector<double*> &extrinsicsVector
) {
	calibration = new double [4];
	calibration[0] = calibrationMatrix.at<double>(0, 0);
	calibration[1] = calibrationMatrix.at<double>(1, 1);
	calibration[2] = calibrationMatrix.at<double>(0, 2);
	calibration[3] = calibrationMatrix.at<double>(1, 2);

	extrinsicsVector.clear();
	for (auto &imageData : imagesDataForAdjustment) {
		double* newExtrinsics = new double[6];
		cv::Mat r;
		cv::Rodrigues(imageData.rotation, r);
		newExtrinsics[0] = r.at<double>(0);
		newExtrinsics[1] = r.at<double>(1);
		newExtrinsics[2] = r.at<double>(2);
		newExtrinsics[3] = imageData.motion.at<double>(0);
		newExtrinsics[4] = imageData.motion.at<double>(1);
		newExtrinsics[5] = imageData.motion.at<double>(2);
		extrinsicsVector.push_back(newExtrinsics);
	}
}

static void convertDataFromBA(
	double* calibration, std::vector<double*> &extrinsicsVector,
	cv::Mat& calibrationMatrix, std::vector<TemporalImageData> &imagesDataForAdjustment
) {
	calibrationMatrix.at<double>(0, 0) = calibration[0];
	calibrationMatrix.at<double>(1, 1) = calibration[0];
	calibrationMatrix.at<double>(0, 2) = calibration[0];
	calibrationMatrix.at<double>(1, 2) = calibration[0];
	delete calibration;

	for (int i = 0; i < extrinsicsVector.size(); ++i) {
		auto &imageData = imagesDataForAdjustment[i];
		double* extrinsics = extrinsicsVector[i];
		cv::Mat r(3, 1, CV_64F);
		r.at<double>(0) = extrinsics[0];
		r.at<double>(1) = extrinsics[1];
		r.at<double>(2) = extrinsics[2];
		cv::Rodrigues(r, imageData.rotation);
		imageData.motion.at<double>(0) = extrinsics[3];
		imageData.motion.at<double>(1) = extrinsics[4];
		imageData.motion.at<double>(2) = extrinsics[5];
		delete extrinsics;
	}
}