#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

int main(int argc, char** argv) {

    Mat image, result;
    for (int i = 0; i < 4; i++) {
        // Читаем изображение, сводя его к оттенкам серого.
        image = imread(format("data/%d.jpg", i), ImreadModes::IMREAD_GRAYSCALE);
        if (!image.data) {
            printf("No image data \n");
            return -1;
        }

        // Создаём детектор.
        Ptr<FastFeatureDetector> detector = 
            FastFeatureDetector::create(14, true, FastFeatureDetector::TYPE_9_16);
        std::vector<KeyPoint> keypoints;

        // Находим фичи.
        detector->detect(image, keypoints);

        // Выделяем фичи.
        drawKeypoints(image, keypoints, result);

        namedWindow("Display Image", WINDOW_AUTOSIZE);
        // Выводим результат.
        imshow("Display Image", result);
        // Каждое изображение показывается 6 секунд.
        waitKey(6000);
    }
    return 0;
}