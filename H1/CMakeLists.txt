add_library(julia utilities.cu utilities.h
        julia_gpu.cu julia_gpu.h)

set_target_properties(julia PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(julia PUBLIC cuda_std_11)

add_executable(h1 main.cpp
        AnimeCreator.cpp AnimeCreator.h
        anime_head.cpp anime_head.h anime_juliaset.cpp anime_juliaset.h anime_tail.cpp anime_tail.h)

target_compile_definitions(h1 PUBLIC H1_JULIA_USE_GPU)
target_link_libraries(h1 PUBLIC julia)
target_link_libraries(h1 PUBLIC ${OpenCV_LIBS})
target_include_directories(h1 PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(h1-cpu main.cpp
        AnimeCreator.cpp AnimeCreator.h
        anime_head.cpp anime_head.h anime_juliaset.cpp anime_juliaset.h anime_tail.cpp anime_tail.h)
target_link_libraries(h1-cpu PUBLIC julia)
target_link_libraries(h1-cpu PUBLIC ${OpenCV_LIBS})