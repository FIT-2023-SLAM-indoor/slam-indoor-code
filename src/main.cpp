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
void convertKeyPointIntoPoints(vector<KeyPoint>* keypoints, vector<Point2f>* points)
{

	for (int i = 0;i < (*keypoints).size();i++)
		(*points).push_back((*keypoints)[i].pt);
}
/*
* Gets the feature and returns a vector of batch with given radius consists of points around this feature.
* Barrier is used to choose the density of resulting circle;
*/
void getPointsAroundFeature(Point2f feature, int radius, double barier, vector<Point2f>* pointsAround)
{
	(*pointsAround).push_back(feature);
	for (double k = 0;k < radius;k++)
	{
		for (double fi = 0;fi < barier;fi++)
		{
			Point2f point;
			point.x = ceil(k * cos(2 * M_PI * fi / barier) + feature.x);

			point.y = ceil(k * sin(2 * M_PI * fi / barier) + feature.y);
			if (std::count((*pointsAround).begin(), (*pointsAround).end(), point))
				continue;
			if (point.x < WIDTH && point.x >= 0 && point.y < HEIGHT && point.y >= 0)
				(*pointsAround).push_back(point);
		}
	}

}
/*
* Gets two circle batches from two different pictures and returs sum of squared difference of pixels colors of given batches.
*/
double sumSquaredDifferences(vector<Point2f>* batch1, vector<Point2f>* batch2, Mat* image1, Mat* image2, double min)
{
	double sum = 0;
	vector<Point2f>* mn;
	vector<Point2f>* mx;
	if (batch1->size() <= batch2->size())
	{
		mn = batch1;
		mx = batch2;
	}

	else {
		mn = batch2;
		mx = batch1;
	}
	for (int i = 0;i < mn->size();i++)
	{
		Vec3b& color1 = image1->at<Vec3b>((*batch1)[i]);
		Vec3b& color2 = image2->at<Vec3b>((*batch2)[i]);
		for (int j = 0;j < 3;j++)
		{
			sum += (color1[j] - color2[j]) * (color1[j] - color2[j]);
		}

		if (sum > min)
			return sum;
	}
	sum += (mx->size() - mn->size()) * (sum / mn->size());

	return sum;
}
/*
* Main func to track features.Gets feature,two images and address of feature in the second image.
*/
int trackFeature(Point2f feature, Mat* image1, Mat* image2, Point2f* res, double barier)
{
	double sigma = (WIDTH + HEIGHT) / (4 * sqrt(HEIGHT + WIDTH));
	int r = ceil(sigma);
	vector<Point2f> circ;
	getPointsAroundFeature(feature, r + 1, barier, &circ);

	double min = INFINITY;
	double sum;


	for (int j = 0;j < circ.size();j++) {
		vector<Point2f> currentCirc;

		getPointsAroundFeature(circ[j], r + 1, barier, &currentCirc);
		sum = sumSquaredDifferences(&circ, &currentCirc, image1, image2, min);
		if (sum < min) {
			min = sum;
			if (circ[j].x < 20 || circ[j].y < 20 || circ[j].x > WIDTH - 20 || circ[j].y > HEIGHT - 20)
				return -1;
			(*res) = circ[j];
		}

		currentCirc.clear();
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
	HEIGHT = image.size().height;
	WIDTH = image.size().width;
	vector<KeyPoint> keypoints;
	vector<Point2f> points;
	fastExtractor(&image, &keypoints, 20);
	original = image.clone();
	convertKeyPointIntoPoints(&keypoints, &points);
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
		original = image2.clone();
		vector<Point2f> newFeatures;

		for (int j = 0;j < points.size();j++) {

			Point2f feature;
			if (trackFeature(points[j], &image, &image2, &feature, 10) == -1)
				continue;
			Vec3b color = image2.at<Vec3b>(feature);
			color[0] = 0;
			color[1] = 0;
			color[2] = 255;
			image2.at<Vec3b>(feature) = color;
			newFeatures.push_back(feature);
		}
		points = newFeatures;
		imshow("Live", image2);
		image = original;

		char c = (char)waitKey(33);
		if (c == 27)
			break;

	}
	return 0;
}