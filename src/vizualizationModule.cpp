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
    window.registerKeyboardCallback(KeyboardViz3d,&window);
    window.setWindowPosition(Point(0,0));
    window.setViewerPose(path[0]);
    window.spin();
    
}
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());
 
    return  norm(I, shouldBeIdentity) < 1e-6;
}
Vec3f rotationMatrixToEulerAngles(Mat &R)
{
 
    assert(isRotationMatrix(R));
 
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
}
void KeyboardViz3d(const viz::KeyboardEvent &w, void *window)
{
    
    viz::Viz3d* ptr = (viz::Viz3d*)window;
    Affine3d affine = ptr->getViewerPose();
    Mat rotation = (Mat)affine.rotation();
    Vec3f eulerAngles = rotationMatrixToEulerAngles(rotation);
    Vec3d past =  affine.translation();

    Vec3d newTranslation;
    double speed = 0.5;
    newTranslation[2] = cos(eulerAngles[1]);
    std::cout << past[0] << std::endl;
    std::cout << eulerAngles[1]*180/3.14159 << " " << cos(eulerAngles[1]) << std::endl;
    

    newTranslation[0] = sin(eulerAngles[1]);
    std::cout << eulerAngles[1]*180/3.14159 << " " << sin(eulerAngles[1]) << std::endl;
  

    if (w.action){
        std::cout << "you pressed "<< w.symbol<< " " << (int)w.code << std::endl;
        switch ((int)w.code){
            case 119: //w 
                std::cout << eulerAngles << std::endl;
                ptr->setViewerPose(affine.translate(newTranslation));
                break;
            case 32: //space
                Vec3d newTranslation;
                newTranslation[1] = -1;
                ptr->setViewerPose(affine.translate(newTranslation));
                break;
  
        }
    }
        
    
}