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
 * @param [out] goodFrame найденный кадр с достаточным количеством ключевых точек
 * @param [out] goodFrameFeatures ключевые точки полученного кадра
 * @return true, если удалось найти изображение с достаточным количеством точек, иначе false
 */
bool findFirstGoodFrame(
    MediaSources &mediaInputStruct,
    const DataProcessingConditions &dataProcessingConditions,
    Mat &goodFrame,
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
 * @param [in] prevFrameData
 * @param [out] newFrameData
 * @param [in, out] keyPointFrameCoords1
 * @param [in, out] keyPointFrameCoords2
 * @param [out] chiralityMask
 */
void computeTransformationAndFilterPoints(
    const DataProcessingConditions &dataProcessingConditions,
    const TemporalImageData &prevFrameData,
    TemporalImageData &newFrameData,
    std::vector<Point2f> &keyPointFrameCoords1,
    std::vector<Point2f> &keyPointFrameCoords2,
    Mat &chiralityMask
);


/**
 * Эта функция нужна для заполнения поля correspondSpatialPointIdx струкутры данных первого
 * и второго изображения из медиа данных. Сначала для обоих кадров мы задаем размер этому полю
 * равный размеру соответствующего вектора фич (т.е. всех фич, полученных с этого кадра).
 * Изначально это поле заполняется значением -1 (т.е. пока что для каждой фичи не опрделен индекс
 * соответствующей трехмерной точки). Потом за линейное время для хороших фичс первого и второго
 * кадра мы вычисляем индекс соответствующего им матча (для начальной пары кадров эти индексы будут
 * равны искомым индексам трехмерных точек).
 *
 * @param [in] chiralityMask
 * @param [in, out] firstFrameData
 * @param [in, out] secondFrameData
 */
void defineFeaturesCorrespondSpatialIndices(
    const Mat &chiralityMask,
    TemporalImageData &firstFrameData,
    TemporalImageData &secondFrameData
);


/**
 * Написать документацию!!!
 */
void getObjAndImgPoints(
    const std::vector<DMatch> &matches,
    const std::vector<int> &correspondSpatialPointIdx,
    const std::vector<Point3f> &spatialPoints,
    const std::vector<KeyPoint> &extractedFeatures,
    std::vector<Point3f> &objPoints,
    std::vector<Point2f> &imgPoints
);


/**
 * Написать документацию!!!
 */
void pushNewSpatialPoints(
	const std::vector<DMatch> &matches,
	std::vector<int> &prevFrameCorrespondingIndices,
	std::vector<int> &currFrameCorrespondingIndices,
	const std::vector<Point3f> &newSpatialPoints,
	std::vector<Point3f> &allSpatialPoints
); 
