cmake_minimum_required(VERSION 3.5)
project(live-stream CXX)

set(CMAKE_C_FLAGS "-pthread -std=gnu99")
set(CMAKE_EXE_LINKER_FLAGS "-pthread")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
add_definitions(-D_GNU_SOURCE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS} -Wall -fprofile-arcs -ftest-coverage -Wno-deprecated-declarations")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -lgcov")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


if(POLICY CMP0146)
   cmake_policy(SET CMP0146 OLD)
endif()

file(GLOB_RECURSE SRCS   
        src/circleQueue.hpp
        src/Decoder.cpp
        src/Decoder.h
        src/main.cpp
        src/Processor.h
        src/Processor.cpp
        src/Pusher.h
        src/Pusher.cpp      
    )



add_executable( ${PROJECT_NAME} ${SRCS} )

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV installed in the system")
    message(STATUS " - Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS " - Libraries: ${OpenCV_LIBRARIES}")
    add_definitions(-DOPEN_CV_INSTALLED)
    include_directories(${OPENCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
else ()
    message(FATAL "Did not find OpenCV in the system, image data is inside RecvContainer as raw data")
endif ()


find_package(FFMPEG REQUIRED)
if (FFMPEG_FOUND)
    message(STATUS "Found FFMPEG installed in the system")
    message(STATUS " - Includes: ${FFMPEG_INCLUDE_DIR}")
    message(STATUS " - Libraries: ${FFMPEG_LIBRARIES}")
    include_directories(${FFMPEG_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBRARIES})
else ()
    message(STATUS "Cannot Find FFMPEG")
endif (FFMPEG_FOUND)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA installed in the system")
    message(STATUS " - Includes: ${CUDA_INCLUDE_DIRS}")
    message(STATUS " - Libraries: ${CUDA_LIBRARIES}")
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}  ${CUDA_LIBRARIES})
else()
    message(STATUS "Cannot Find CUDA")
endif(CUDA_FOUND)


set(TensorRT_INCLUDE_DIRS  "/usr/include/x86_64-linux-gnu" )
set(TensorRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

include_directories(${TensorRT_INCLUDE_DIRS})
link_directories(${TensorRT_LIB_DIR})

target_link_libraries(${PROJECT_NAME} m stdc++ pthread nvinfer nvinfer_plugin)