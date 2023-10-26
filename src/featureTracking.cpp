#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "featureTracking.h"

using namespace cv;

/*
* Gets the feature and returns a std::vector of batch with given radius consists of points around this feature.
* Barrier is used to choose the density of resulting circle;
*/
void getPointsAroundFeature(Point2f feature, int radius, double barier, std::vector<Point2f>& pointsAround, Mat& img)
{
	int WIDTH = img.size().width;
	int HEIGHT = img.size().height;
	pointsAround.push_back(feature);
	for (double k = 0;k < radius;k++)
	{
		for (double fi = 0;fi < barier;fi++)
		{
			Point2f point;
			point.x = ceil(k * cos(2 * M_PI * fi / barier) + feature.x);

			point.y = ceil(k * sin(2 * M_PI * fi / barier) + feature.y);
			if (std::count(pointsAround.begin(), pointsAround.end(), point))
				continue;
			if (point.x < WIDTH && point.x >= 0 && point.y < HEIGHT && point.y >= 0)
				pointsAround.push_back(point);
		}
	}

}
/*
* Gets two circle batches from two different pictures and returs sum of squared difference of pixels colors of given batches.
*/
double sumSquaredDifferences(std::vector<Point2f>& batch1, std::vector<Point2f>& batch2, Mat& image1, Mat& image2, double min)
{
	double sum = 0;
	std::vector<Point2f> mn;
	std::vector<Point2f> mx;
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
		sum += (color1[0] - color2[0]) * (color1[0] - color2[0]);

		if (sum > min)
			return sum;
	}

	return sum;
}
/*
* Tracking one feature.Gets feature,two images and address of feature in the second image to put the result in int.
* @param barier is required for choosing density of circles.
*
*/
int trackFeature(Point2f feature, Mat& image1, Mat& image2, Point2f& res, double barier, double maxAcceptableDifference)
{
	int WIDTH = image1.size().width;
	int HEIGHT = image1.size().height;
	double sigma = (WIDTH + HEIGHT) / (4 * sqrt(HEIGHT + WIDTH));
	int r = ceil(sigma);
	std::vector<Point2f> circ;
	getPointsAroundFeature(feature, r + 1, barier, circ, image1);

	double min = INFINITY;
	double sum;


	for (int j = 0;j < circ.size();j++) {
		std::vector<Point2f> currentCirc;

		getPointsAroundFeature(circ[j], r + 1, barier, currentCirc, image1);
		sum = sumSquaredDifferences(circ, currentCirc, image1, image2, min);
		if (sum < min) {
			min = sum;
			//if (circ[j].x < r - 1 || circ[j].y < r - 1 || circ[j].x > WIDTH - r + 1 || circ[j].y > HEIGHT - r + 1)
			//	continue;
			res = circ[j];
		}


		currentCirc.clear();
	}
	if (min > maxAcceptableDifference) {
		return -1;
	}
	return 0;
}
/*
* Tracking all the features.
* @param features vector of previous features.
* @param newFeatures vector of tracked features
* @param barier is required to choose density of circles.
*
*/
void trackFeatures(std::vector<Point2f>& features, Mat& previousFrame, Mat& currentFrame, std::vector<Point2f>& newFeatures, int barier, double maxAcceptableDifference) {
	for (int j = 0;j < features.size();j++) {
		Point2f feature;
		if (trackFeature(features[j], previousFrame, currentFrame, feature, barier, maxAcceptableDifference) == -1) {
			features.erase(features.begin() + j);
			j--;
			continue;
		}

		newFeatures.push_back(feature);
	}
}