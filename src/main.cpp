#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;

int main(int argc, char** argv )
{
    VideoCapture cap("data/example.mp4");
    Mat image;
    if (!cap.isOpened()) {
        std::cerr << "Camera wasn't opened" << std::endl;
        return -1;
    }
    while (true) {
        cap.read(image);
        if (image.empty()) {
            std::cerr << "Empty frame" << std::endl;
            return -1;
        }
        imshow("Live", image);
        char c = (char)waitKey(33);
        if (c == 27)
            break;
    }
    return 0;
}