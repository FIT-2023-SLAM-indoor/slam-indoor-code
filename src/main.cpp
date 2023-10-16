#include <opencv2/opencv.hpp>

#include "fastExtractor.h"

using namespace cv;


int main(int argc, char** argv)
{
    Mat image, result;
    // Let's do a little slide show. Let's see how the algorithm processes 4 different images
    for (int i = 0; i < 9; i++)
    {
        // Saved the i-th image into an N-dimensional array
        image = imread(format("data/%d.jpg", i));  // ImreadModes::IMREAD_GRAYSCALE
        if (!image.data) {
            printf("No image data \n");
            return -1;
        }

        std::vector<KeyPoint> keypoints;

        // Applied the FAST algorithm to the image and saved the image
        // with the highlighted features in @result
        fastExtractor(&image, &keypoints, 13);

        drawKeypoints(image, keypoints, result);

        namedWindow("Display Image", WINDOW_AUTOSIZE);
        imshow("Display Image", result);
        // Each image displays for 6 seconds
        waitKey(6000);
    }
    return 0;
}