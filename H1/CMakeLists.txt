add_executable(h1 main.cpp AnimeCreator.cpp AnimeCreator.h)

target_link_libraries(h1 PUBLIC ${OpenCV_LIBS})
target_include_directories(h1 PUBLIC ${OpenCV_INCLUDE_DIRS})

