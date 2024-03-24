#pragma once
#include "fstream"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;

/**
 * This structure is designed to handle both video and photo data sources,
 * allowing the same functions to process either type of media without the need for overriding.
*/
struct MediaSources {
    VideoCapture frameSequence;
    std::vector<String> photosPaths;
    bool isPhotoProcessing;
};

/**
 * Struct containing temporal image data.
 */
struct TemporalImageData {
    std::vector<KeyPoint> allExtractedFeatures;       // All extracted features of the i-th frame.
    std::vector<Vec3b> colorsForAllExtractedFeatures; // Colors for all extracted features of the i-th frame.
    std::vector<DMatch> allMatches;                   // Matches between features of frames i-1 and i.
    Mat rotation;                                     // Rotation between frames i-1 and i.
    Mat motion;                                       // Motion between frames i-1 and i.
    std::vector<int> correspondSpatialPointIdx;       // Labels field for three-dimensional points.
};

struct GlobalData {
	std::vector<Point3f> spatialPoints;
	std::vector<Vec3b> spatialPointsColors;
	std::vector<Mat> spatialCameraPositions;
    std::vector<Mat> cameraRotations;
};


/**
 * Главный цикл обработки медиа данных. Вычисление трехмерных координат ключевых точек изображений
 * будет идти до тех, пока не встретится большое количество подряд идущих кадров, которые тяжело
 * связать с предыдущими обработанными кадрами.
 *
 * @param [in, out] temporalImageDeque
 * @param [out] globalDataStruct 
 * @return 
*/
void mainCycle(std::deque<TemporalImageData> temporalImageDataDeque, GlobalData globalDataStruct);