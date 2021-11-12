cmake_minimum_required(VERSION 3.20)
project(CV_course)

set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_DIR "C:\\Program Files\\opencv\\build")
set(OpenCV_DIR "C:\\Program Files\\opencv\\build\\x64\\vc15\\lib")
find_package(OpenCV REQUIRED)

add_subdirectory(H1)
