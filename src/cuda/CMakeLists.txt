cmake_minimum_required(VERSION 3.23)
project(multirotor_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(cuda_benchmark benchmark.cu)

target_link_libraries(cuda_benchmark
        rl_tools::minimal
        rl_tools::optimizations_cuda
        learning_to_fly_in_seconds
)
RL_TOOLS_TAG_IS_CUDA(cuda_benchmark)
