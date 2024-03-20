#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include "vizualizationModule.h"
using namespace cv;
void vizualizePoints(std::vector<Point3f> spatialPoints)
{
    viz::Viz3d window("Coordinate Frame");
    window.setWindowSize(Size(1000,1000));
    window.setBackgroundColor(); // black by default
    std::vector<Vec3f> point_cloud_est;
    for (int i = 0; i < spatialPoints.size(); ++i)
        point_cloud_est.push_back(Vec3f(spatialPoints[i]));
    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);
    window.setWindowPosition(Point(0,0));
    window.spin();
}
void vizualizePointsAndCameras(
    std::vector<Point3f> spatialPoints,
    std::vector<Mat> rotations, 
    std::vector<Mat> transitions, 
    Mat calibration
    )
{
  viz::Viz3d window("Coordinate Frame");
    window.setWindowSize(Size(1000,1000));

    window.setBackgroundColor(); // black by default
    std::vector<Vec3f> point_cloud_est;
    for (int i = 0; i < spatialPoints.size(); ++i)
        point_cloud_est.push_back(Vec3f(spatialPoints[i]));
    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);

    std::vector<Affine3d> path;
    for (size_t i = 0; i < rotations.size(); ++i)
        path.push_back(Affine3d(rotations[i],transitions[i]));

    cv::Matx33f K((float*)calibration.ptr());

    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path,
    K, 0.1, viz::Color::yellow()));

    window.setWindowPosition(Point(0,0));
    window.setViewerPose(path[0]);
    window.spin();
}
