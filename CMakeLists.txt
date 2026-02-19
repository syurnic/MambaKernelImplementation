cmake_minimum_required(VERSION 4.1)
project(MambaFromScratch CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(MambaFromScratch main.cu
        cuda_executor.cu)

set_target_properties(MambaFromScratch PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
