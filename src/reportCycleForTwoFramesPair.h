#pragma once
#include <opencv2/calib3d.hpp>

/**
 * Main cycle test with frames-pairs.
 *
 * @param [in] FEATURE_EXTRACTING_THRESHOLD
 * @param [in] FEATURE_TRACKING_BARRIER
 * @param [in] FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE
 */
void reportingCycleForFramesPairs(int FEATURE_EXTRACTING_THRESHOLD, int FEATURE_TRACKING_BARRIER,
                                  int FEATURE_TRACKING_MAX_ACCEPTABLE_DIFFERENCE);