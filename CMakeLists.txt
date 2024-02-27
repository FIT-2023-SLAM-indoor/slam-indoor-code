cmake_minimum_required(VERSION 3.0.0)
project(slam-indoor-code VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#find_package( Ceres REQUIRED  )

add_executable(
        slam-indoor-code
        "src/main.cpp"
        "src/main_config.h"
        "src/fastExtractor.cpp" "src/fastExtractor.h"
        "src/cameraCalibration.cpp" "src/cameraCalibration.h"
        "src/cameraTransition.cpp" "src/cameraTransition.h"
        "src/featureTracking.cpp" "src/featureTracking.h"
        "src/triangulate.cpp" "src/triangulate.h"
        "src/videoProcessingCycle.h" "src/videoProcessingCycle.cpp"
        "src/photosProcessingCycle.cpp" "src/photosProcessingCycle.h"
        "src/bundleAdjustment.cpp" "src/bundleAdjustment.h"
        "src/IOmisc.cpp" "src/IOmisc.h"
)

target_link_libraries( slam-indoor-code ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
 