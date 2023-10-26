#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

#include "fastExtractor.h"
#include "featureTracking.h"
using namespace cv;


int main(int argc, char** argv)
{
	Mat image, image2, original;
	VideoCapture cap("data/example.mp4");
	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}
	cap.read(image);
	cvtColor(image, image, COLOR_BGR2GRAY);
	cvtColor(image, image, COLOR_GRAY2BGR);
	std::vector<KeyPoint> keypoints;
	std::vector<Point2f> points;
	fastExtractor(image, keypoints, 20);
	original = image.clone();
	KeyPoint::convert(keypoints, points);
	for (int i = 0;i < points.size();i++)
	{
		Vec3b color = original.at<Vec3b>(points[i]);
		color[0] = 0;
		color[1] = 0;
		color[2] = 255;
		original.at<Vec3b>(points[i]) = color;
	}

	while (true) {
		cap.read(image2);
		if (image2.empty()) {
			std::cerr << "Empty frame" << std::endl;
			return -1;
		}
		cvtColor(image2, image2, COLOR_BGR2GRAY);
		cvtColor(image2, image2, COLOR_GRAY2BGR);
		original = image2.clone();

		std::vector<Point2f> newFeatures;

		trackFeatures(points, image, image2, newFeatures, 10, 10000);
		std::cout << points.size() << "  " << newFeatures.size() << "\n";
		points = newFeatures;
		for (int j = 0;j < newFeatures.size();j++) {

			Vec3b color = image2.at<Vec3b>(newFeatures[j]);
			color[0] = 0;
			color[1] = 0;
			color[2] = 255;
			image2.at<Vec3b>(newFeatures[j]) = color;
		}
		imshow("Live", image2);
		image = original;

		char c = (char)waitKey(33);
		if (c == 27)
			break;

	}
	return 0;
}