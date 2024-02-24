#pragma once

#include "main_config.h"
#include <opencv2/videoio.hpp>

using namespace cv;

/**
 * Sorts files photos paths by its numbers.
 *
 * @param paths string names' vector
 */
void sortGlobs(std::vector<String>& paths);