add_subdirectory(juliaset)

add_executable(hw1 main.cpp
        AnimeCreator.cpp AnimeCreator.h
        anime_head.cpp anime_head.h anime_juliaset.cpp anime_juliaset.h anime_tail.cpp anime_tail.h)
target_compile_definitions(hw1 PUBLIC H1_JULIA_USE_GPU)
target_link_libraries(hw1 PUBLIC julia-gpu)
target_link_libraries(hw1 PUBLIC ${OpenCV_LIBS})
target_include_directories(hw1 PUBLIC ${OpenCV_INCLUDE_DIRS})
target_compile_features(hw1 PRIVATE cuda_std_11)
add_custom_command(TARGET hw1
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/imgs" "$<TARGET_FILE_DIR:hw1>/imgs")

add_executable(hw1-cpu main.cpp
        AnimeCreator.cpp AnimeCreator.h
        anime_head.cpp anime_head.h anime_juliaset.cpp anime_juliaset.h anime_tail.cpp anime_tail.h)
target_link_libraries(hw1-cpu PUBLIC julia-cpu)
target_link_libraries(hw1-cpu PUBLIC ${OpenCV_LIBS})
target_include_directories(hw1-cpu PUBLIC ${OpenCV_INCLUDE_DIRS})
target_compile_features(hw1-cpu PRIVATE cxx_std_11)
add_custom_command(TARGET hw1-cpu
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/imgs" "$<TARGET_FILE_DIR:hw1-cpu>/imgs")