cmake_minimum_required(VERSION 3.0.0)
project(slam-indoor-code VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( Ceres REQUIRED  )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(nlohmann_json 3.10.0 REQUIRED)

add_executable(
    slam-indoor-code
    src/main.cpp

    src/mainModule/cycleProcessing/mainCycleStructures.h
    src/mainModule/cycleProcessing/mainCycle.cpp src/mainModule/cycleProcessing/mainCycle.h
    src/mainModule/cycleProcessing/batch.cpp src/mainModule/cycleProcessing/batch.h
    src/mainModule/cycleProcessing/mainCycleInternals.cpp src/mainModule/cycleProcessing/mainCycleInternals.h

    src/mainModule/featureExtraction/fastExtractor.cpp src/mainModule/featureExtraction/fastExtractor.h

    src/calibration/cameraCalibration.cpp src/calibration/cameraCalibration.h

    src/mainModule/translation/cameraTranslation.cpp src/mainModule/translation/cameraTranslation.h

    src/mainModule/featureMatching/featureMatchingCommon.cpp src/mainModule/featureMatching/featureMatchingCommon.h
    src/mainModule/featureMatching/featureMatching.h

    src/mainModule/triangulation/triangulate.cpp src/mainModule/triangulation/triangulate.h

    src/mainModule/bundleAdjustment/bundleAdjustment.cpp src/mainModule/bundleAdjustment/bundleAdjustment.h

    src/vizualization/vizualizationModule.cpp src/vizualization/vizualizationModule.h

    src/vizualization/delauney-triangulation/bestFittingPlane.cpp src/vizualization/delauney-triangulation/bestFittingPlane.h
    src/vizualization/delauney-triangulation/geomAdditionalFunc.cpp src/vizualization/delauney-triangulation/geomAdditionalFunc.h
    src/vizualization/delauney-triangulation/bowyerWatson.cpp src/vizualization/delauney-triangulation/bowyerWatson.h

    src/config/config.cpp src/config/ConfigService.h src/config/configData.h src/config/config.h

    src/misc/IOmisc.cpp src/misc/IOmisc.h
    src/misc/ChronoTimer.cpp src/misc/ChronoTimer.h

)

if (DEFINED USE_CUDA)
    add_definitions(-DUSE_CUDA)
    target_sources(
        slam-indoor-code
        PRIVATE src/mainModule/featureMatching/featureMatchingCUDA.cpp
    )
else()
    target_sources(
        slam-indoor-code
        PRIVATE src/mainModule/featureMatching/featureMatchingCPU.cpp
    )
endif ()

target_link_libraries( slam-indoor-code Ceres::ceres )
target_link_libraries( slam-indoor-code ${OpenCV_LIBS} )
target_link_libraries( slam-indoor-code nlohmann_json::nlohmann_json )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
 