add_executable(hw2-train train.cpp FaceLoader.cpp FaceLoader.h)
target_link_libraries(hw2-train PRIVATE ${OpenCV_LIBS})
target_include_directories(hw2-train PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(hw2-test test.cpp FaceLoader.cpp FaceLoader.h)
target_link_libraries(hw2-test PRIVATE ${OpenCV_LIBS})
target_include_directories(hw2-test PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(hw2-reconstruct reconstruct.cpp FaceLoader.cpp FaceLoader.h)
target_link_libraries(hw2-reconstruct PRIVATE ${OpenCV_LIBS})
target_include_directories(hw2-reconstruct PRIVATE ${OpenCV_INCLUDE_DIRS})
