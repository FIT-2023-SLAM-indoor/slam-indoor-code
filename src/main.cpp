#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

/* Gaussian filter */

/*
 *
 */
void fast_extractor(Mat* src_image, Mat* result, int threshold = 10, bool suppression = true,
    FastFeatureDetector::DetectorType type = FastFeatureDetector::TYPE_9_16)
{
    //
    Ptr<FastFeatureDetector> detector =
        FastFeatureDetector::create(threshold, suppression, type);

    //
    std::vector<KeyPoint> keypoints;

    //
    detector->detect(*src_image, keypoints);

    //
    drawKeypoints(*src_image, keypoints, *result);
}

int main(int argc, char** argv)
{
    Mat image, result;
    // Let's do a little slide show. Let's see how the algorithm processes 4 different images
    for (int i = 0; i < 4; i++)
    {
        // Saved the i-th image into an N-dimensional array
        image = imread(format("data/%d.jpg", i), ImreadModes::IMREAD_GRAYSCALE);  // ImreadModes::IMREAD_GRAYSCALE
        if (!image.data) {
            printf("No image data \n");
            return -1;
        }

        // Applied the FAST algorithm to the image and saved the image
        // with the highlighted features in @result
        fast_extractor(&image, &result, 13);

        namedWindow("Display Image", WINDOW_AUTOSIZE);
        imshow("Display Image", result);
        // Each image displays for 6 seconds
        waitKey(6000);
    }
    return 0;
}