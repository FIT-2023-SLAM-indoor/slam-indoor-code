#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <thread>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "featureTracking.h"

#include "main_config.h"
#define STANDART_FT
#define THREADS_COUNT 10;
using namespace cv;


void getPointsAroundFeature(Point2f feature, int radius, double barier, std::vector<Point2f>& pointsAround, Mat& img)
{
	int WIDTH = img.size().width;
	int HEIGHT = img.size().height;
	pointsAround.push_back(feature);
	for (double k = 0; k < radius; k++)
	{
		for (double fi = 0; fi < barier; fi++)
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

	else
	{
		mn = batch2;
		mx = batch1;
	}
	for (int i = 0; i < mn.size(); i++)
	{
		Vec3b& color1 = image1.at<Vec3b>(batch1[i]);
		Vec3b& color2 = image2.at<Vec3b>(batch2[i]);
		sum += abs(color1[0] - color2[0]);

		if (sum > min)
			return sum;
	}

	return sum;
}

void trackFeature(Point2f feature, Mat& image1, Mat& image2, Point2f& res, double barier, double maxAcceptableDifference)
{
	int WIDTH = image1.size().width;
	int HEIGHT = image1.size().height;
	double sigma = (WIDTH + HEIGHT) / (4 * sqrt(HEIGHT + WIDTH));
	int r = ceil(sigma);
	std::vector<Point2f> circ;
	getPointsAroundFeature(feature, r + 1, barier, circ, image1);

	double min = INFINITY;
	double sum;

	for (int j = 0; j < circ.size(); j++)
	{
		std::vector<Point2f> currentCirc;

		getPointsAroundFeature(circ[j], r + 1, barier, currentCirc, image1);
		sum = sumSquaredDifferences(circ, currentCirc, image1, image2, min);
		if (sum < min)
		{
			min = sum;
			// if (circ[j].x < r - 1 || circ[j].y < r - 1 || circ[j].x > WIDTH - r + 1 || circ[j].y > HEIGHT - r + 1)
			//	continue;
			res = circ[j];
		}

		currentCirc.clear();
	}
	if (min > maxAcceptableDifference)
	{
		res.x = -1;
	}

}
void function(Point2f feature, Mat& image1, Mat& image2, Point2f& res, double barier, double maxAcceptableDifference)
{
	trackFeature(feature, image1, image2, res, barier, maxAcceptableDifference);
}

void trackFeatures(std::vector<Point2f>& features, Mat& previousFrame, Mat& currentFrame, std::vector<Point2f>& newFeatures, int barier, double maxAcceptableDifference)
{
#ifdef STANDART_FT

	std::vector<Point2f> isGoodFeatures;
	for (int i = 0;i < features.size();i++) {
		isGoodFeatures.push_back(Point2f());
	}
	for (int j = 0; j < features.size(); j++)
	{
		std::thread th1(function, features[j], std::ref(previousFrame),
			std::ref(currentFrame), std::ref(isGoodFeatures.at(j)), barier, maxAcceptableDifference);
		j++;
		if (j < features.size()) {
			std::thread th2(function, features[j], std::ref(previousFrame),
				std::ref(currentFrame), std::ref(isGoodFeatures.at(j)), barier, maxAcceptableDifference);
			j++;
			if (j < features.size()) {
				std::thread th3(function, features[j], std::ref(previousFrame),
					std::ref(currentFrame), std::ref(isGoodFeatures.at(j)), barier, maxAcceptableDifference);
				th3.join();
				j++;
			}
			th2.join();

		}
		
		th1.join();
		
		
	}
	int deletedCount = 0;
	for (int i = 0;i < isGoodFeatures.size();i++) {
		if (isGoodFeatures.at(i).x == -1) {
			features.erase(features.begin() + (i - deletedCount));
		    deletedCount++;
		}
		else {
			newFeatures.push_back(isGoodFeatures.at(i));
		}
	}
	isGoodFeatures.clear();
#else
	int WIDTH = previousFrame.size().width;
	int HEIGHT = previousFrame.size().height;
	std::vector<uchar> status;
	std::vector<float> err;
	Size winSize = Size(21, 21);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 0.01);

	calcOpticalFlowPyrLK(previousFrame, currentFrame, features, newFeatures, status, err, winSize, 3, termcrit, 0, 0.001);

	//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	int indexCorrection = 0;
	for (int i = 0; i < status.size(); i++)
	{

		Point2f pt = newFeatures.at(i - indexCorrection);
		if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0) || (pt.x >= WIDTH - 3) || (pt.y >= HEIGHT - 3)) {
			if ((pt.x < 0) || (pt.y < 0) || (pt.x >= WIDTH - 3) || (pt.y >= HEIGHT - 3)) {
				status.at(i) = 0;
			}

			features.erase(features.begin() + (i - indexCorrection));
			newFeatures.erase(newFeatures.begin() + (i - indexCorrection));
			indexCorrection++;
		}

	}

#endif
}