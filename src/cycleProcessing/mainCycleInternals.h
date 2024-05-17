#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "mainCycleStructures.h"

using namespace cv;

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
 * Определяет поля позиции камеры в TemporalImageData. Функция универсально работает с начальным
 * кадром и последующими.
 *
 * @param [in] oldImageDataDeque
 * @param [in] lastFrameOfLaunchId
 * @param [out] frameData
 */
void defineCameraPosition(
    const std::deque<TemporalImageData> &oldImageDataDeque, 
    int lastFrameOfLaunchId, 
    TemporalImageData &frameData
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
 * @param [in] calibrationMatrix
 * @param [in] firstFrameData
 * @param [out] secondFrameData
 * @param [in, out] keyPointFrameCoords1
 * @param [in, out] keyPointFrameCoords2
 * @param [out] chiralityMask
 */
void computeTransformationAndFilterPoints(
    const DataProcessingConditions &dataProcessingConditions,
	Mat &calibrationMatrix,
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
    const SpatialPointsVector &allSpatialPoints,
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
    const SpatialPointsVector &newSpatialPoints,
	GlobalData &globalDataStruct,
	std::vector<int> &prevFrameCorrespondIndices,
	TemporalImageData &newFrameData
);


/**
 * Функция копирует общие данные об обработанном медиа из newGlobalData в mainGlobalData.
 *
 * @param [out] mainGlobalData
 * @param [in] newGlobalData
 */
void insertNewGlobalData(GlobalData &mainGlobalData, GlobalData &newGlobalData);


/**
 * Функция проверяет была ли заполнена структура GlobalData после обработки медиа. Если структура
 * осталась пуста, то будет вызвано падение программы.
 *
 * @param [in] globalDataStruct
 */
void checkGlobalDataStruct(GlobalData &globalDataStruct);