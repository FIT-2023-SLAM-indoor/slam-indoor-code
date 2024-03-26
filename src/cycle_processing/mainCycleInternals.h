#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;

/**
 * Struct containing conditions for data processing.
 */
struct DataProcessingConditions {
    Mat calibrationMatrix;            // Calibration matrix for camera.
    Mat distortionCoeffs;             // Distortion coefficients for camera.
    int frameBatchSize;               // Size of batch of frames.
    int featureExtractingThreshold;   // Threshold for feature extraction.
    int requiredExtractedPointsCount; // Required number of extracted points in frame.
    int requiredMatchedPointsCount;   // Required number of matched points in frame.
    int matcherType;                  // Type of descriptor matcher to use.
};


/**
 * Функция устонавливает значение предусловиям обработки медиа данных.
 *
 * @param [out] mediaInputStruct так называемый интерефейс для универсальной работы с медиа
 * @param [out] dataProcessingConditions
 */
void defineProcessingEnvironment(
    MediaSources &mediaInputStruct, 
    DataProcessingConditions &dataProcessingConditions
);


/**
 * Функция достает (и удаляет) из структуры слдующее изображение.
 * Новое изображение сохрахраняется в nextFrame.
 *
 * @param [in, out] mediaInputStruct из структуры вынимется очередное изображение
 * @param [out] nextFrame 
 * @return true, если удалось достать непустое изображение, иначе false
 */
bool getNextFrame(MediaSources &mediaInputStruct, Mat &nextFrame);


/**
 * Функция вынимает изображения из потока медиа данных, пока не будет найдено хорошее
 * изображение с нужным количеством ключевых точек или пока не закроется поток.
 *
 * @param [in, out] mediaInputStruct из структуры вынимются изображения
 * @param [in] dataProcessingConditions
 * @param [out] firstGoodFrame найденный кадр с достаточным количеством ключевых точек
 * @param [out] goodFrameFeatures ключевые точки полученного кадра
 * @return true, если удалось найти изображение с достаточным количеством точек, иначе false
 */
bool findFirstGoodFrame(
    MediaSources &mediaInputStruct,
    const DataProcessingConditions &dataProcessingConditions,
    Mat &firstGoodFrame,
    std::vector<KeyPoint> &goodFrameFeatures
);


/**
 * Функция на основе данных (векторов фич и вектора матчей) первых двух кадров вычисляет координаты
 * каждой ключевой точки на первом и втором кадре. Эти метаданные помогают апроксимировать
 * поворот и сдвиг камеры при переходе от первого ко второму кадру. Вместе с этим будет получена
 * матрица хиральности, с помощью которой будут отфильтрованы ключевые точки и соответствующие
 * координаты не попадут в выходные вектора.
 *
 * @param [in] dataProcessingConditions
 * @param [in] firstFrameData
 * @param [out] secondFrameData
 * @param [in, out] keyPointFrameCoords1
 * @param [in, out] keyPointFrameCoords2
 * @param [out] chiralityMask
 */
void computeTransformationAndFilterPoints(
    const DataProcessingConditions &dataProcessingConditions,
    const TemporalImageData &firstFrameData,
    TemporalImageData &secondFrameData,
    std::vector<Point2f> &keyPointFrameCoords1,
    std::vector<Point2f> &keyPointFrameCoords2,
    Mat &chiralityMask
);


/**
 * Эта функция нужна для заполнения поля correspondSpatialPointIdx струкутры данных первого
 * и второго изображения из медиа данных. Сначала для обоих кадров мы задаем размер этому полю
 * равный размеру соответствующего вектора фич (т.е. всех фич, полученных с этого кадра).
 * Изначально это поле заполняется значением -1 (т.е. пока что для каждой фичи не опрделен индекс
 * соответствующей трехмерной точки). Потом за линейное время для хороших фич из первого и второго
 * кадра мы вычисляем индекс соответствующего им матча (для начальной пары кадров эти индексы будут
 * равны искомым индексам трехмерных точек).
 * Подобным образом сохранятся цвет из первого изображения для соответствующей трехмерной точки.
 *
 * @param [in] chiralityMask
 * @param [in] secondFrame
 * @param [in, out] firstFrameData
 * @param [in, out] secondFrameData
 * @param [out] firstPairSpatialPointColors
 */
void defineFeaturesCorrespondSpatialIndices(
    const Mat &chiralityMask, 
    const Mat &secondFrame, 
    TemporalImageData &firstFrameData, 
    TemporalImageData &secondFrameData,
    std::vector<Vec3b> &firstPairSpatialPointColors
);


/**
 * Функция определяет, какие трехмерные точки, из вычисленных для предыдущих кадров, соответствуют
 * фичам для нового кадра. В качестве выходных значений получаются трехмерные точки и координаты
 * их фич на изображении нового кадра.
 *
 * @param [in] matches
 * @param [in] prevFrameCorrespondIndices
 * @param [in] allSpatialPoints
 * @param [in] newFrameKeyPoints
 * @param [out] oldSpatialPointsForNewFrame
 * @param [out] newFrameFeatureCoords
 */
void getOldSpatialPointsAndNewFrameFeatureCoords(
    const std::vector<DMatch> &matches,
    const std::vector<int> &prevFrameCorrespondIndices,
    const std::vector<Point3f> &allSpatialPoints,
    const std::vector<KeyPoint> &newFrameKeyPoints,
    std::vector<Point3f> &oldSpatialPointsForNewFrame,
    std::vector<Point2f> &newFrameFeatureCoords
);


/**
 * Благодаря этой функции получаем из трехмерных точек вычисленных для нового кадра только те
 * трехмерные точки, которые не получены из предыдущих кадров (т.е. это новые трехмерные точки).
 * А для уже существующих трехмерных точек просто записываем соответствующие индексы в поле
 * newFrameCorrespondIndices структуры для нового кадра.
 *
 * @param [in] newFrame
 * @param [in] newSpatialPoints
 * @param [out] globalDataStruct
 * @param [in, out] prevFrameCorrespondIndices
 * @param [in, out] newFrameData WE NEED ONLY ...
 */
void pushNewSpatialPoints(
    const Mat &newFrame,
    const std::vector<Point3f> &newSpatialPoints,
	GlobalData &globalDataStruct,
	std::vector<int> &prevFrameCorrespondIndices,
	TemporalImageData &newFrameData
);


/**
 * Сохраняем цвет трехмерной точки, получая из кадра цвет фичи соответствующего матча.
 *
 * @param [in] frame
 * @param [in] matchIdx ABOBA!!!
 * @param [in, out] frameData
 * @param [out] spatialPointColors
 */
void saveFrameColorOfKeyPoint(
    const Mat &frame, int matchIdx, 
    TemporalImageData &frameData, 
    std::vector<Vec3b> &spatialPointColors
);
