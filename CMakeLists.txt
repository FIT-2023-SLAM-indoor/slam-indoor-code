cmake_minimum_required(VERSION 3.0.0)
project(slam-indoor-code VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(
        slam-indoor-code "src/main.cpp"
        "src/fastExtractor.cpp" "src/fastExtractor.h"
        "src/cameraCalibration.cpp" "src/cameraCalibration.h"
        "src/cameraTransition.cpp" "src/cameraTransition.h"
        "src/featureTracking.cpp" "src/featureTracking.h"
        "src/triangulate.cpp" "src/triangulate.h"
        "src/videoProcessingCycle.h" "src/videoProcessingCycle.cpp"
        "src/reportCycleForTwoFramesPair.cpp" "src/reportCycleForTwoFramesPair.h"
        "src/main_config.h"
        "src/photosProcessingCycle.cpp" "src/photosProcessingCycle.h")

target_link_libraries( slam-indoor-code ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
 