cmake_minimum_required(VERSION 3.20)
project(CV_course CXX C CUDA)

set(CMAKE_CXX_STANDARD 11)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
    set(CMAKE_C_FLAGS "/EHa /EHs")
    set(CMAKE_CXX_FLAGS "/EHa /EHs")
endif ()

find_package(OpenCV CONFIG REQUIRED)

add_subdirectory(HW1)
add_subdirectory(HW2)
