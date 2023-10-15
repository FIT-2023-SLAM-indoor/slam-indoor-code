#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

int main(int argc, char** argv) {

    Mat image, result;
    for (int i = 0; i < 4; i++) {
        // ������ �����������, ����� ��� � �������� ������.
        image = imread(format("data/%d.jpg", i), ImreadModes::IMREAD_GRAYSCALE);
        if (!image.data) {
            printf("No image data \n");
            return -1;
        }

        // ������ ��������.
        Ptr<FastFeatureDetector> detector = 
            FastFeatureDetector::create(14, true, FastFeatureDetector::TYPE_9_16);
        std::vector<KeyPoint> keypoints;

        // ������� ����.
        detector->detect(image, keypoints);

        // �������� ����.
        drawKeypoints(image, keypoints, result);

        namedWindow("Display Image", WINDOW_AUTOSIZE);
        // ������� ���������.
        imshow("Display Image", result);
        // ������ ����������� ������������ 6 ������.
        waitKey(6000);
    }
    return 0;
}