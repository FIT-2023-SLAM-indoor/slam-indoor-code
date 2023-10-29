#include "opencv2/core/core_c.h"

#include "triangulate.h"

constexpr int NUM_VIEWS = 2;
constexpr int PROJ_MATR_COLS = 4;


/*
 *
 * 
 * 
 */
static void reconstructPointsFor3D(CvMat* projMatr1, CvMat* projMatr2, CvMat* projPoints1, CvMat* projPoints2, CvMat* points4D)
{
    int numPoints = projPoints1->cols;

    // preallocate SVD matrices on stack
    cv::Matx<double, 4, 4> matrA;
    cv::Matx<double, 4, 4> matrU;
    cv::Matx<double, 4, 1> matrW;
    cv::Matx<double, 4, 4> matrV;

    CvMat* projPoints[2] = { projPoints1, projPoints2 };
    CvMat* projMatrs[2] = { projMatr1, projMatr2 };

    // Solve system for each point
    for (int i = 0; i < numPoints; i++)    // For each point 
    {
        // Fill matrix for current point
        for (int j = 0; j < NUM_VIEWS; j++)    // For each view
        {
            double x, y;
            x = cvmGet(projPoints[j], 0, i);
            y = cvmGet(projPoints[j], 1, i);
            for (int k = 0; k < PROJ_MATR_COLS; k++)
            {
                matrA(j * 2, k) = x * cvmGet(projMatrs[j], 2, k) - cvmGet(projMatrs[j], 0, k);
                matrA(j * 2 + 1, k) = y * cvmGet(projMatrs[j], 2, k) - cvmGet(projMatrs[j], 1, k);
            }
        }
        // Solve system for current point
        cv::SVD::compute(matrA, matrW, matrU, matrV);

        // Copy computed point
        cvmSet(points4D, 0, i, matrV(3, 0));/* X */
        cvmSet(points4D, 1, i, matrV(3, 1));/* Y */
        cvmSet(points4D, 2, i, matrV(3, 2));/* Z */
        cvmSet(points4D, 3, i, matrV(3, 3));/* W */
    }
}


/*
 *
 * 
 * 
 */
void triangulate(cv::InputArray projPoints1, cv::InputArray projPoints2,
    cv::Mat matr1, cv::Mat matr2,
    cv::OutputArray points4D)
{
    cv::Mat points1 = projPoints1.getMat(), points2 = projPoints2.getMat();

    CvMat cvMatr1 = cvMat(matr1), cvMatr2 = cvMat(matr2);
    CvMat cvPoints1 = cvMat(points1), cvPoints2 = cvMat(points2);

    //
    points4D.create(4, points1.cols, points1.type());    // 
    cv::Mat matPoints4D = points4D.getMat();
    CvMat cvPoints4D = cvMat(matPoints4D);
    
    reconstructPointsFor3D(&cvMatr1, &cvMatr2, &cvPoints1, &cvPoints2, &cvPoints4D);
}