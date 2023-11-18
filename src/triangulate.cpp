#include <iostream>

#include "opencv2/calib3d.hpp"
#include "opencv2/core/core_c.h"

#include "triangulate.h"

using namespace cv;

constexpr int NUM_VIEWS = 2;
constexpr int PROJ_MATR_COLS = 4;

/**
 * This function reconstructs 3-dimensional points (in homogeneous coordinates).
 * Calculates matrix A, after that denotes three-dimensional positions for all points.
 * To do this, we use the singular value decomposition (SVD) over A.
 * For more details about linear triangulation see the book "Multiple View Geometry in CV".
 */
static void reconstructPointsFor3D(CvMat& projMatr1, CvMat& projMatr2, CvMat& projPoints1, CvMat& projPoints2, CvMat& points4D)
{
    int numPoints = projPoints1.rows;

    // Preallocate SVD matrices on stack.
    cv::Matx<double, 4, 4> matrA;
    cv::Matx<double, 4, 4> matrU;
    cv::Matx<double, 4, 1> matrW;
    cv::Matx<double, 4, 4> matrV;

    CvMat* projPoints[2] = { &projPoints1, &projPoints2 };
    CvMat* projMatrs[2] = { &projMatr1, &projMatr2 };

    // Solve system for each point.
    for (int p = 0; p < numPoints; p++)   // For each Point. 
    {
        // Fill matrix for current point.
        for (int v = 0; v < NUM_VIEWS; v++)   // For each View.
        {
            double x, y;
            x = cvmGet(projPoints[v], p, 0);
            y = cvmGet(projPoints[v], p, 1);
            for (int c = 0; c < PROJ_MATR_COLS; c++)   // For each Column.
            {
                matrA(v * 2, c) = x * cvmGet(projMatrs[v], 2, c) - cvmGet(projMatrs[v], 0, c);
                matrA(v * 2 + 1, c) = y * cvmGet(projMatrs[v], 2, c) - cvmGet(projMatrs[v], 1, c);
            }
        }
        // Solve system for current point.
        cv::SVD::compute(matrA, matrW, matrU, matrV);

        // Write computed point into points array.
        cvmSet(&points4D, 0, p, matrV(3, 0)); /* X */
        cvmSet(&points4D, 1, p, matrV(3, 1)); /* Y */
        cvmSet(&points4D, 2, p, matrV(3, 2)); /* Z */
        cvmSet(&points4D, 3, p, matrV(3, 3)); /* W */
        // TODO: in the future, you can swap (p) and column indexes, and set the value not of columns, but of rows at once
    }
}

void triangulate(cv::InputArray projPoints1, cv::InputArray projPoints2,
    const cv::Mat& matr1, const cv::Mat& matr2,
    cv::OutputArray points4D)
{
    cv::Mat points1 = projPoints1.getMat(), points2 = projPoints2.getMat();

    CvMat cvMatr1 = cvMat(matr1), cvMatr2 = cvMat(matr2);
    CvMat cvPoints1 = cvMat(points1), cvPoints2 = cvMat(points2);

    // Create the array for our 3D points. 
    points4D.create(4, points1.rows, points1.type());   // Four rows because we have additional parameter W for coords.
    cv::Mat matPoints4D = points4D.getMat();
    CvMat cvPoints4D = cvMat(matPoints4D);

    reconstructPointsFor3D(cvMatr1, cvMatr2, cvPoints1, cvPoints2, cvPoints4D);
}

void convertPointsFromHomogeneousWrapper(cv::Mat& homogeneous3DPoints, cv::Mat& euclideanPoints)
{
    homogeneous3DPoints = homogeneous3DPoints.t(); // TODO: This line must be deleted when you will process points as matrix Nx4
//    cv::convertPointsFromHomogeneous(homogeneous3DPoints, euclideanPoints); OpenCV version with strange matrix sizes
    for (int row = 0; row < homogeneous3DPoints.rows; ++row) {
        homogeneous3DPoints.row(row) /= homogeneous3DPoints.at<double>(row, 3);
    }
    euclideanPoints = (homogeneous3DPoints.colRange(0, homogeneous3DPoints.cols-1)).clone();
}

void placeEuclideanPointsInWorldSystem(Mat& points, Mat& worldCameraPose, Mat& worldCameraRotation)
{
//    worldEuclideanPoints.create(points.rows, points.cols, CV_64F);
    Mat point, rotatedPoint;
    for (int r = 0; r < points.rows; ++r) {
        point = points.row(r).clone();
        point = point.t();
        rotatedPoint = worldCameraRotation * point;
        rotatedPoint = rotatedPoint.t();
        point = worldCameraPose + rotatedPoint;
        points.row(r) = point.clone();
    }
}