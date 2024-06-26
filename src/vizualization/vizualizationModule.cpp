#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include "../config/config.h"
#include "vizualizationModule.h"
#include "delauney-triangulation/geomAdditionalFunc.h"
#include "delauney-triangulation/bestFittingPlane.h"

using namespace cv;

double speed = 0.5;

void vizualizeOnlyPoints(
    std::vector<Point3f>& spatialPoints,
    std::vector<Vec3b>& colors)
{
    viz::Viz3d window = makeWindow();
    viz::WCloud cloudWidget = getPointCloudFromPoints(spatialPoints,colors);
    window.showWidget("point_cloud", cloudWidget);
}

viz::WCloud getPointCloudFromPoints(
    std::vector<Point3f>& spatialPoints,
    std::vector<Vec3b>& colors)
{
    std::vector<Vec3f> point_cloud_est;
    for (int i = 0; i < spatialPoints.size(); ++i)
        point_cloud_est.push_back(Vec3f(spatialPoints[i]));
    if (colors.size()  == spatialPoints.size()){
        viz::WCloud cloud_widget(point_cloud_est, colors);
        return cloud_widget;
    }
    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    return cloud_widget;
}

viz::Viz3d makeWindow()
{
    viz::Viz3d window("Coordinate Frame");
    window.setWindowSize(Size(1000, 1000));
    window.setBackgroundColor(viz::Color(255, 255, 255));
    return window;
}

void vizualizeCameras(
    viz::Viz3d& window,
    std::vector<Mat>& rotations,
    std::vector<Mat>& transitions, 
    Mat& calibration)
{
    std::vector<Affine3d> path;
    for (size_t i = 0; i < rotations.size(); ++i)
        path.push_back(Affine3d(rotations[i], transitions[i]));
    cv::Matx33f K((float *)calibration.ptr());
    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
//    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path,K, 0.1, viz::Color::yellow()));
    Point3f centroid(-0.368855, 0.538447, 7.92543);
    window.setViewerPose(Affine3d(rotations[0],Mat(centroid)));

}

void vizualizePointsAndCameras(
    std::vector<Point3f>& spatialPoints,
    std::vector<Mat>& rotations,
    std::vector<Mat>& transitions,
    std::vector<Vec3b>& colors,
    Mat& calibration)
{
    viz::Viz3d window = makeWindow();
    viz::WCloud cloudWidget = getPointCloudFromPoints(spatialPoints,colors);
    window.showWidget("point_cloud", cloudWidget);
    
    std::vector<std::vector<int>> comps;


    
	clusterizePoints(spatialPoints,colors,comps);
    

    Vec3d normal;
    Point3f centroid;
    vizualizeCameras(window,rotations,transitions,calibration);
    getBestFittingPlaneByPoints(spatialPoints,centroid,normal);
    cout<< "normal: " <<endl;
    cout<< normal  <<endl;
    cout<< "centroid:" << centroid << endl;
    
   

    

    
    
    for (int i =0;i< comps.size();i++){
        std::vector<Point3f> compPoints;
        std::vector<Vec3b> compColors;

        if (comps[i].size() < configService.getValue<int>(ConfigFieldEnum::TRIANGLE_MINIMUM_TRIANGLE_POINTS))
            continue;
        
        for (int j = 0;j< comps[i].size();j++){
            int index = comps[i].at(j);
            compPoints.push_back(spatialPoints.at(index));
            compColors.push_back(colors.at(index));
        }
        try{
            
            cv::viz::WMesh trWidget = makeMesh(compPoints,compColors,centroid,normal);
            std::string s = std::to_string(i);
            char const *pchar = s.c_str();
            window.showWidget(pchar,trWidget);

        } catch (const std::exception& e) 
        {
            std::cout << e.what(); 
        }
        
        
        compPoints.clear();
        compColors.clear();
       
    }
    
    
   

    
    
    /*
    viz::WPlane bestFittingPlane(centroid,normal,Vec3d(1,1,1),Size2d(Point2d(150,150)));
    //window.showWidget("plane", bestFittingPlane);
    */
    
    startWindowSpin(window);
}

void startWindowSpin(
    viz::Viz3d& window)
{
    window.registerKeyboardCallback(KeyboardViz3d, &window);
    window.setWindowPosition(Point(0, 0));
    window.spinOnce();
    viz::Camera cam = viz::Camera(Vec2d(1.0,1.0),cv::Size(1000,1000));
    window.setCamera(cam);
    std::cout << cam.getFov() << std::endl;
    std::cout << window.getCamera().getFov() << std::endl;
    window.spinOnce(3600000,true);
}

Vec3f rotationMatrixToEulerAngles(
    Mat &rotationMatrix)
{
    cv::Mat euler(3, 1, CV_64F);
    double m00 = rotationMatrix.at<double>(0, 0);
    double m02 = rotationMatrix.at<double>(0, 2);
    double m10 = rotationMatrix.at<double>(1, 0);
    double m11 = rotationMatrix.at<double>(1, 1);
    double m12 = rotationMatrix.at<double>(1, 2);
    double m20 = rotationMatrix.at<double>(2, 0);
    double m22 = rotationMatrix.at<double>(2, 2);
    double bank, attitude, heading;
    // Assuming the angles are in radians.
    if (m10 > 0.998)
    { // singularity at north pole
        bank = 0;
        attitude = CV_PI / 2;
        heading = atan2(m02, m22);
    }
    else if (m10 < -0.998)
    { // singularity at south pole
        bank = 0;
        attitude = -CV_PI / 2;
        heading = atan2(m02, m22);
    }
    else
    {
        bank = atan2(-m12, m11);
        attitude = asin(m10);
        heading = atan2(-m20, m00);
    }
    euler.at<double>(0) = bank;
    euler.at<double>(1) = attitude;
    euler.at<double>(2) = heading;

    return euler;
}

void KeyboardViz3d(const viz::KeyboardEvent &w, void *window)
{

    viz::Viz3d *ptr = (viz::Viz3d *)window;
    Affine3d affine = ptr->getViewerPose();
    Mat rotation = (Mat)affine.rotation();
    Vec3f eulerAngles = rotationMatrixToEulerAngles(rotation);
    Vec3d past = affine.translation();

    viz::Camera cam = ptr->getCamera();

    //std::cout << "Angle 2 :" << eulerAngles[2] * 180 / 3.14159 << "  Cos:" << cos(eulerAngles[2]) << std::endl;

    //std::cout << "Angle 2 :" << eulerAngles[2] * 180 / 3.14159 << " Sin:" << sin(eulerAngles[2]) << std::endl;

    //std::cout << eulerAngles << std::endl;
    Vec3d newTranslation;
    if (w.action)
    {
        //std::cout <<  cam.getFov() << std::endl;
        //std::cout << "you pressed " << w.symbol << " " << (int)w.code << std::endl;
        switch ((int)w.code)
        {
        case 119: // w
            newTranslation[2] = cos(eulerAngles[2]) * speed;
            newTranslation[0] = sin(eulerAngles[2]) * speed;
            ptr->setViewerPose(affine.translate(newTranslation));
            break;
        case 115: // s
            newTranslation[2] = -cos(eulerAngles[2]) * speed;
            newTranslation[0] = -sin(eulerAngles[2]) * speed;
            ptr->setViewerPose(affine.translate(newTranslation));
            break;
        case 97: // a
            newTranslation[2] = cos(eulerAngles[2] - 3.14 / 2.0) * speed;
            newTranslation[0] = sin(eulerAngles[2] - 3.14 / 2.0) * speed;
            ptr->setViewerPose(affine.translate(newTranslation));
            break;
        case 100: // d
            newTranslation[2] = cos(eulerAngles[2] + 3.14 / 2.0) * speed;
            newTranslation[0] = sin(eulerAngles[2] + 3.14 / 2.0) * speed;
            ptr->setViewerPose(affine.translate(newTranslation));
            break;
        case 99: // c
            newTranslation[1] = speed * speed;
            ptr->setViewerPose(affine.translate(newTranslation));
            break;
        case 61: //+
            if (speed < 2.5)
                speed += 0.25;
            std::cout << "Current speed" << speed << std::endl;
            break;
        case 45: // -
            if (speed > 0.25)
                speed -= 0.25;
            std::cout << "Current speed" << speed << std::endl;
            break;
        case 32: // space
            newTranslation[1] = -speed * speed;
            ptr->setViewerPose(affine.translate(newTranslation));
            break;
        }
    }
}