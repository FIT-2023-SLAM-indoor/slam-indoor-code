#pragma once
#include "fstream"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "mainCycleStructures.h"

using namespace cv;

/**
 * Главный цикл обработки медиа данных. Вычисление трехмерных координат ключевых точек изображений
 * будет идти до тех, пока не встретится большое количество подряд идущих кадров, которые тяжело
 * связать с предыдущими обработанными кадрами.
 *
 * @param [in, out] temporalImageDeque
 * @param [out] globalDataStruct 
 * @return IN PROGRESS
 */
bool mainCycle(
	MediaSources &mediaInputStruct, const DataProcessingConditions &dataProcessingConditions,
	std::deque<TemporalImageData> &temporalImageDataDeque, GlobalData &globalDataStruct
);
