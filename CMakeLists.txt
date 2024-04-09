cmake_minimum_required(VERSION 3.0.0)
project(slam-indoor-code VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( Ceres REQUIRED  )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(nlohmann_json 3.10.0 REQUIRED)

add_executable(
    slam-indoor-code
    src/main.cpp
    src/cycleProcessing/mainCycleStructures.h
    src/cycleProcessing/mainCycle.cpp src/cycleProcessing/mainCycle.h
    src/cycleProcessing/mainCycleInternals.cpp src/cycleProcessing/mainCycleInternals.h
    src/fastExtractor.cpp src/fastExtractor.h
    src/cameraCalibration.cpp src/cameraCalibration.h
    src/cameraTransition.cpp src/cameraTransition.h
    src/featureMatching.cpp src/featureMatching.h
    src/triangulate.cpp src/triangulate.h
    src/bundleAdjustment/bundleAdjustment.cpp src/bundleAdjustment/bundleAdjustment.h
    src/IOmisc.cpp src/IOmisc.h
    src/vizualizationModule.cpp src/vizualizationModule.h
    src/config/config.cpp src/config/ConfigService.h src/config/configData.h src/config/config.h
)

target_link_libraries( slam-indoor-code Ceres::ceres )
target_link_libraries( slam-indoor-code ${OpenCV_LIBS} )
target_link_libraries( slam-indoor-code nlohmann_json::nlohmann_json )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
 