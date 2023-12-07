#pragma once

int photosProcessingCycle(std::vector<String> &photosPaths, int featureTrackingBarier, int featureTrackingMaxAcceptableDiff,
                          int framesBatchSize, int requiredExtractedPointsCount, int featureExtractingThreshold, char* reportsDirPath);