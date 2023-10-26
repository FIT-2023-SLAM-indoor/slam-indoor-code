#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include "fastExtractor.h"

using namespace cv;
using namespace std;

int HEIGHT;
int WIDTH;
/*
* Takes keypoints vector and return points vector.
* The main purpose is to get the spatial coordinates of keypoints and make vector of them.
*
*/
void convertKeyPointIntoPoints(vector<KeyPoint>& keypoints, vector<Point2f>& points)
{

	for (int i = 0;i < keypoints.size();i++)
		points.push_back(keypoints[i].pt);
}
/*
* Gets the feature and returns a vector of batch with given radius consists of points around this feature.
* Barrier is used to choose the density of resulting circle;
*/
int getPointsAroundFeature(Point2f feature, int radius, Mat& image, Mat& pointsAround)
{
	Mat mask = Mat::zeros(image.size(), image.type());
	Mat maskedImg = Mat::zeros(image.size(), image.type());
	cv::circle(mask, cv::Point(feature.x, feature.y), radius, cv::Scalar(255, 255, 255), -1, 8, 0);
	image.copyTo(maskedImg, mask);
	Rect rectangleMask;
	rectangleMask.x = feature.x - radius;
	rectangleMask.y = feature.y - radius;
	rectangleMask.width = 2 * radius + 1;
	rectangleMask.height = 2 * radius + 1;
	if (rectangleMask.x < 0 || rectangleMask.y < 0 || rectangleMask.x > WIDTH - 2 * radius - 2 || rectangleMask.y > HEIGHT - 2 * radius - 2)
		return -1;
	pointsAround = maskedImg(rectangleMask);
	return 0;
}


/*
* Gets two circle batches from two different pictures and returs sum of squared difference of pixels colors of given batches.
*/
double sumSquaredDifferences(vector<Point2f>& batch1, vector<Point2f>& batch2, Mat& image1, Mat& image2, double min)
{
	double sum = 0;
	vector<Point2f> mn;
	vector<Point2f> mx;
	if (batch1.size() <= batch2.size())
	{
		mn = batch1;
		mx = batch2;
	}

	else {
		mn = batch2;
		mx = batch1;
	}
	for (int i = 0;i < mn.size();i++)
	{
		Vec3b& color1 = image1.at<Vec3b>(batch1[i]);
		Vec3b& color2 = image2.at<Vec3b>(batch2[i]);
		for (int j = 0;j < 3;j++)
		{
			sum += (color1[j] - color2[j]) * (color1[j] - color2[j]);
		}

		if (sum > min)
			return sum;
	}
	sum += (mx.size() - mn.size()) * (sum / mn.size());

	return sum;
}
double sumSquaredDifferencesOptimized(Point2f feature, int radius, Mat& initialMask, Mat& image2, double min)
{
	double sm = 0;
	Mat circ;
	if (getPointsAroundFeature(feature, (initialMask.size().width - 1) / 2, image2, circ) == -1)
		return -1;
	Mat dif;
	absdiff(circ, initialMask, dif);
	sm = sum(dif)[0];

	return sm;
}
/*
* Main func to track features.Gets feature,two images and address of feature in the second image.
*/

int trackFeature(Point2f feature, Mat& image1, Mat& image2, Point2f& res, double barier)
{
	double sigma = (WIDTH + HEIGHT) / (4 * sqrt(HEIGHT + WIDTH));
	int r = ceil(sigma);
	Mat circ;
	if (getPointsAroundFeature(feature, r, image1, circ) == -1)
		return -1;

	double min = INFINITY;
	double sum;
	for (int x = 0;x < circ.size().width;x++)
	{
		for (int y = 0;y < circ.size().width;y++) {
			Vec3b& color1 = circ.at<Vec3b>(y, x);
			if (color1[0] == 0)
				continue;
			Point2f curPoint = feature;
			curPoint.x = feature.x - r + x;
			curPoint.y = feature.y - r + y;
			Mat curBatch;
			sum = sumSquaredDifferencesOptimized(curPoint, r, circ, image2, min);
			if (sum == -1)
				return -1;
			if (sum < min) {
				res = curPoint;
				min = sum;
			}

		}

	}



	return 0;
}
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
	original = image.clone();
	HEIGHT = image.size().height;
	WIDTH = image.size().width;
	vector<KeyPoint> keypoints;
	vector<Point2f> points;
	fastExtractor(&image, &keypoints, 30);

	convertKeyPointIntoPoints(keypoints, points);
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
		vector<Point2f> newFeatures;

		for (int j = 0;j < points.size();j++) {

			Point2f feature;
			if (trackFeature(points[j], image, image2, feature, 10) == -1) {
				continue;
			}
			Vec3b color = image2.at<Vec3b>(feature);
			color[0] = 0;
			color[1] = 0;
			color[2] = 255;
			image2.at<Vec3b>(feature) = color;
			newFeatures.push_back(feature);
		}
		/*cout << "\n\nfeatures\n";
		for (int i = 0;i < points.size();i++)
			cout << points[i].x << "," << points[i].y << ",";
		cout << "\n\nnew_features\n";
		for (int i = 0;i < newFeatures.size();i++)
			cout << newFeatures[i].x << "," << newFeatures[i].y << ",";
		*/
		points = newFeatures;

		imshow("Live", image2);
		image = original;

		char c = (char)waitKey(33);
		if (c == 27)
			break;

	}

	return 0;
}