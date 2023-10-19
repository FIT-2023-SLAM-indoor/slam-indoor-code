#include <opencv2/opencv.hpp>

#include "fastExtractor.h"

using namespace cv;


int main(int argc, char** argv)
{
    Mat image, result;
    std::vector<KeyPoint> features;
    // Let's do a little slide show. Let's see how the algorithm processes 7 different images
    for (int i = 0; i < 7; i++)
    {
        // Saved the i-th image into an N-dimensional array
        image = imread(format("data/%d.jpg", i));  // ImreadModes::IMREAD_GRAYSCALE
        if (!image.data) {
            printf("No image data \n");
            return -1;
        }

        // Applied the FAST algorithm to the image and saved the image
        // with the highlighted features in @result
        fastExtractor(image, features);

        drawKeypoints(image, features, result);

        namedWindow("Display Image", WINDOW_AUTOSIZE);
        imshow("Display Image", result);
        // Each image displays for 3 seconds
        waitKey(3000);
    }
    return 0;
}