#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cstdio>

#include "fastExtractor.h"
#include "featureTracking.h"
#include "cameraTransition.h"
#define ESC_KEY 27

using namespace cv;


int main(int argc, char** argv)
{
	// Feature extracting
	////////////////////////////////////////
	Mat image, image2, result;
	std::vector<KeyPoint> keypoints;

	// Saved the image into an N-dimensional array
	VideoCapture cap("data/indoor_test.mp4");  // ImreadModes::IMREAD_GRAYSCALE

	if (!cap.isOpened()) {
		std::cerr << "Camera wasn't opened" << std::endl;
		return -1;
	}
	Mat prevP, currentP;
	while (true) {
		cap.read(image);
		cvtColor(image, image, COLOR_BGR2GRAY);
		cvtColor(image, image, COLOR_GRAY2BGR);

		// Applied the FAST algorithm to the image and saved the image
		// with the highlighted features in @result
		fastExtractor(image, keypoints, 30);
		drawKeypoints(image, keypoints, result);

		namedWindow("Display Image", WINDOW_AUTOSIZE);
		imshow("Display Image", result);
		// Each image displays for 4 seconds
		waitKey(1000);
		// 
		//Transform image into black and white
		cap.read(image2);
		cvtColor(image2, image2, COLOR_BGR2GRAY);
		cvtColor(image2, image2, COLOR_GRAY2BGR);

		std::vector<Point2f> features;
		std::vector<Point2f> newFeatures;
		KeyPoint::convert(keypoints, features);

		trackFeatures(features, image, image2, newFeatures, 10, 10000);

		//Getting keypoints vector to show from points vector(needed only for afcts, you can delete it)
		std::vector<KeyPoint> keypoints2;
		for (size_t i = 0; i < newFeatures.size(); i++) {
			keypoints2.push_back(cv::KeyPoint(newFeatures[i], 1.f));
		}

		// Second image with tracked features
		drawKeypoints(image2, keypoints2, result);
		imshow("Display Image", result);
		waitKey(1000);

		Mat q = Mat(features);
		Mat g = Mat(newFeatures);
		//q = (cv::Mat_<double>(148, 2) << 626, 74, 1112, 107, 882, 115, 908, 191, 916, 201, 910, 202, 989, 217, 985, 250, 987, 253, 970, 261, 970, 271, 989, 273, 625, 276, 718, 277, 721, 279, 702, 299, 954, 299, 956, 299, 989, 304, 998, 304, 709, 305, 978, 309, 685, 314, 972, 314, 995, 314, 987, 315, 911, 316, 982, 318, 997, 319, 1257, 320, 980, 321, 1248, 322, 992, 323, 1255, 324, 1255, 326, 926, 328, 1255, 329, 975, 330, 681, 332, 1255, 332, 421, 334, 930, 336, 1258, 337, 608, 339, 972, 339, 927, 344, 975, 350, 1005, 351, 910, 353, 980, 354, 992, 354, 706, 357, 981, 358, 731, 360, 1004, 361, 822, 363, 976, 365, 985, 367, 1001, 367, 997, 368, 1008, 370, 791, 371, 1012, 371, 989, 374, 640, 380, 980, 380, 911, 383, 976, 383, 772, 394, 990, 395, 972, 397, 1019, 397, 1074, 397, 1005, 398, 995, 400, 1008, 400, 1020, 401, 995, 403, 1016, 408, 1075, 425, 450, 428, 1083, 429, 1074, 430, 1030, 437, 1033, 438, 1060, 443, 1083, 443, 923, 444, 449, 448, 978, 455, 401, 462, 926, 466, 1089, 469, 1061, 478, 1082, 479, 983, 488, 1151, 502, 1157, 502, 1151, 504, 1153, 509, 984, 514, 1154, 522, 1163, 522, 1165, 522, 982, 523, 1104, 524, 1160, 525, 1165, 529, 1162, 531, 1009, 532, 1167, 532, 1030, 536, 1167, 537, 1024, 549, 948, 550, 1158, 550, 1017, 557, 948, 565, 1136, 565, 937, 566, 972, 566, 985, 566, 1138, 567, 827, 569, 956, 573, 999, 576, 1086, 576, 1028, 587, 1125, 591, 1127, 591, 1142, 597, 1139, 598, 1134, 599, 1133, 601, 973, 603, 1154, 603, 994, 607, 1005, 611, 499, 612, 507, 613, 1003, 640, 1140, 641, 1144, 641, 1135, 654, 1125, 691, 1104, 695, 1122, 695, 1125, 697);
		//g = (cv::Mat_<double>(148, 2) << 626, 76, 1112, 109, 882, 117, 908, 193, 916, 203, 910, 204, 989, 219, 985, 252, 987, 255, 970, 263, 970, 273, 989, 275, 625, 278, 718, 279, 721, 281, 702, 301, 954, 301, 956, 301, 989, 306, 998, 306, 709, 307, 978, 311, 685, 316, 972, 316, 995, 316, 987, 317, 911, 318, 982, 320, 997, 321, 1257, 322, 980, 323, 1248, 324, 992, 325, 1255, 326, 1255, 328, 926, 330, 1255, 331, 975, 332, 681, 334, 1255, 334, 421, 336, 930, 338, 1258, 339, 608, 341, 972, 341, 927, 346, 975, 352, 1005, 353, 910, 355, 980, 356, 992, 356, 706, 359, 981, 360, 731, 362, 1004, 363, 822, 365, 976, 367, 985, 369, 1001, 369, 997, 370, 1008, 372, 791, 373, 1012, 373, 989, 376, 640, 382, 980, 382, 911, 385, 976, 385, 772, 396, 990, 397, 972, 399, 1019, 399, 1074, 399, 1005, 400, 995, 402, 1008, 402, 1020, 403, 995, 405, 1016, 410, 1075, 427, 450, 430, 1083, 431, 1074, 432, 1030, 439, 1033, 440, 1060, 445, 1083, 445, 923, 446, 449, 450, 978, 457, 401, 464, 926, 468, 1089, 471, 1061, 480, 1082, 481, 983, 490, 1151, 504, 1157, 504, 1151, 506, 1153, 511, 984, 516, 1154, 524, 1163, 524, 1165, 524, 982, 525, 1104, 526, 1160, 527, 1165, 531, 1162, 533, 1009, 534, 1167, 534, 1030, 538, 1167, 539, 1024, 551, 948, 552, 1158, 552, 1017, 559, 948, 567, 1136, 567, 937, 568, 972, 568, 985, 568, 1138, 569, 827, 571, 956, 575, 999, 578, 1086, 578, 1028, 589, 1125, 593, 1127, 593, 1142, 599, 1139, 600, 1134, 601, 1133, 603, 973, 605, 1154, 605, 994, 609, 1005, 613, 499, 614, 507, 616, 1003, 642, 1140, 643, 1144, 643, 1135, 656, 1125, 693, 1104, 697, 1122, 697, 1125, 699);
		q = q.reshape(1);
		g = g.reshape(1);

		////////////////////////////////////////
		// Estimate matrices
		////////////////////////////////////////
		currentP = Mat(3, 4, CV_32F);
		countMatrices(q, g, currentP);
		////////////////////////////////////////

		///something with P matrix...
		//Probably here we skip step if this is the first two frames

		//////////////////////////////////////
		char c = (char)waitKey(33);
		if (c == ESC_KEY)
			break;
		prevP = currentP;

	}



	return 0;
}