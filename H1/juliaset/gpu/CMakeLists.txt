add_library(julia-gpu utilities.cu utilities.h
        julia_gpu.cu julia_gpu.h)

set_target_properties(julia-gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(julia-gpu PUBLIC cuda_std_11)