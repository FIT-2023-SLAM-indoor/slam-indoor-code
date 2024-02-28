#pragma once
#include "fstream"

#include <opencv2/core.hpp>

using namespace cv;

/**
 *
 */
typedef struct LogFilesStreams {
    std::ofstream mainReportStream;
    std::ofstream pointsStream;
    std::ofstream poseStream;
    std::ofstream poseHandyStream;
    std::ofstream poseGlobalMltStream;
} LogFilesStreams;

/**
 *
 */
typedef struct MainData {
    int capacity; // Максимальная размерность векторов
    int size; /// количество кадров, инфа о которых есть сейчас

    std::vector<std::vector<KeyPoint>> allFeatures; // Все заэкстратенные фичи i-го кадра
    std::vector<std::vector<Vec3b>> colorsForFeatures; // цвета j-й фичи i-го кадра
    std::vector<std::vector<DMatch>> allMatches; // Матчи между фичами кадров i и i+1
    std::vector<Point3f> worldPoints; // Трёхмерные точки
    std::vector<Vec3b> colors; // Цвета трёхмерных точек
    std::vector<std::vector<int>> correspondWorldPointIdx; // Поле меток для трёхмерных точек (см. ниже)
    std::vector<Mat> rotations; // Вращения между кадрами i-1 и i
    std::vector<Mat> motions; // Сдвиги между кадрами i-1 и i
} MainDataStructure;
