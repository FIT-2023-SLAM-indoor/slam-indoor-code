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
    Matx<double, 4, 4> matrA;
    Matx<double, 4, 4> matrU;
    Matx<double, 4, 1> matrW;
    Matx<double, 4, 4> matrV;

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
        SVD::compute(matrA, matrW, matrU, matrV);

        // Write computed point into points array.
        cvmSet(&points4D, 0, p, matrV(3, 0)); /* X */
        cvmSet(&points4D, 1, p, matrV(3, 1)); /* Y */
        cvmSet(&points4D, 2, p, matrV(3, 2)); /* Z */
        cvmSet(&points4D, 3, p, matrV(3, 3)); /* W */
        // TODO: in the future, you can swap (p) and column indexes, and set the value not of columns, but of rows at once
    }
}

void triangulationWrapper(InputArray projPoints1, InputArray projPoints2,
						  const Mat& matr1, const Mat& matr2,
						  OutputArray points4D)
{
    Mat points1 = projPoints1.getMat(), points2 = projPoints2.getMat();

    CvMat cvMatr1 = cvMat(matr1), cvMatr2 = cvMat(matr2);
    CvMat cvPoints1 = cvMat(points1), cvPoints2 = cvMat(points2);

    // Create the array for our 3D points. 
    points4D.create(4, points1.rows, points1.type());   // Four rows because we have additional parameter W for coords.
    Mat matPoints4D = points4D.getMat();
    CvMat cvPoints4D = cvMat(matPoints4D);

    reconstructPointsFor3D(cvMatr1, cvMatr2, cvPoints1, cvPoints2, cvPoints4D);
}

void reconstruct(
	const Mat& calibration,
	const Mat& rotation1, const Mat& transition1,
	const Mat& rotation2, const Mat& transition2,
	const std::vector<Point2f>& points1, const std::vector<Point2f>& points2,
	std::vector<Point3f>& spatialPoints
) {
	Mat projection1(3, 4, CV_64F);
	Mat projection2(3, 4, CV_64F);
	hconcat(rotation1, transition1, projection1);
	hconcat(rotation2, transition2, projection2);

	projection1 = calibration * projection1;
	projection2 = calibration * projection2;

	Mat homogeneousSpatialPoints;
	// If there will be types' errors, convert vectors to matrices manually
	Mat pointsMat1 = Mat(points1);
	pointsMat1.reshape(1).convertTo(pointsMat1, CV_64F);
	Mat pointsMat2 = Mat(points2);
	pointsMat2.reshape(1).convertTo(pointsMat2, CV_64F);
	triangulationWrapper(pointsMat1, pointsMat2, projection1, projection2, homogeneousSpatialPoints);

	convertHomogeneousPointsMatrixToSpatialPointsVector(homogeneousSpatialPoints, spatialPoints);
}

void convertHomogeneousPointsMatrixToSpatialPointsVector(
	const Mat& homogeneous3DPoints, std::vector<Point3f>& spatialPoints
)
{
	spatialPoints.clear();
	double w;
	Mat pointCol = Mat::zeros(4, 1, CV_64F);
	for (int col = 0; col < homogeneous3DPoints.cols; ++col) {
		w = homogeneous3DPoints.at<double>(3, col);
		pointCol = homogeneous3DPoints.col(col);
		pointCol /= w;
		spatialPoints.emplace_back(
			pointCol.at<double>(0),
			pointCol.at<double>(1),
			pointCol.at<double>(2)
		);
	}
}

void normalizeHomogeneousWrapper(const Mat& inputHomogeneous3DPoints, Mat& normalizedHomogeneous3DPoints)
{
//    Mat homogeneous3DPoints = inputHomogeneous3DPoints.t();
//    convertPointsFromHomogeneous(homogeneous3DPoints, euclideanPoints); OpenCV version with strange matrix sizes
    int normalizedColIndex = 0;
    double w;
    Mat pointCol = Mat::zeros(4, 1, CV_64F);
    normalizedHomogeneous3DPoints.create(4, inputHomogeneous3DPoints.cols, CV_64F);
    for (int col = 0; col < inputHomogeneous3DPoints.cols; ++col) {
        w = inputHomogeneous3DPoints.at<double>(3, col);
        if (fabs(w) < 0.000000000001) {
            continue;
        }
        pointCol = inputHomogeneous3DPoints.col(col).clone();
        pointCol /= w;
        pointCol.copyTo(normalizedHomogeneous3DPoints.col(normalizedColIndex++));
    }
    normalizedHomogeneous3DPoints = normalizedHomogeneous3DPoints.colRange(0, normalizedColIndex).clone();
}

void placeEuclideanPointsInWorldSystem(Mat& points, Mat& worldCameraPose, Mat& worldCameraRotation)
{
//    worldEuclideanPoints.create(points.rows, points.cols, CV_64F);
    Mat point;
    points = worldCameraRotation * points;
    for (int column = 0; column < points.cols; ++column) {
        point = points.col(column).clone();
        point += worldCameraPose;
        points.col(column) = point.clone();
    }
}