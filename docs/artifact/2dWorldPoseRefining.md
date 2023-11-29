# Code below can be use for checking pose refining in two dimensions

```c
#include <iostream>
#define _USE_MATH_DEFINES

#include "cameraTransition.h"
#include "videoProcessingCycle.h"

using namespace cv;

#define ESC_KEY 27
#define FEATURE_EXTRACTING_THRESHOLD 10
#define FEATURE_TRACKING_BARRIER 10
#define FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE 10000


#define FRAMES_GAP 2
#define REQUIRED_EXTRACTED_POINTS_COUNT 10

#define DEBUG

int main(int argc, char** argv)
{
#ifdef DEBUG
    Mat worldCameraPose = Mat::zeros(1, 2, CV_64F),
            rotation(2, 2, CV_64F),
            transition(2, 1, CV_64F);
    double x = 0, y = 0, angle = 0;
    scanf("%lf %lf %lf", &x, &y, &angle);
    while (x > -0.000000001) {
        rotation.at<double>(0, 0) = cos(M_PI * angle / 180);
        rotation.at<double>(0, 1) = -sin(M_PI * angle / 180);
        rotation.at<double>(1, 0) = sin(M_PI * angle / 180);
        rotation.at<double>(1, 1) = cos(M_PI * angle / 180);
        transition.at<double>(0, 0) = x;
        transition.at<double>(1, 0) = y;
        worldCameraPose += (rotation * transition).t();
        std::cout << worldCameraPose << std::endl;
        scanf("%lf %lf %lf", &x, &y, &angle);
    }
    return 0;
}
```