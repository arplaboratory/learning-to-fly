# CUDA Quadrotor Simulator Test


This is supposed to be configured from the parentt directory using CMake
```
mkdir build
cd build
CUDA_PATH=/usr/local/cuda-12.3/ cmake .. -DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build .
./src/cuda/cuda_benchmark
```

Note that the parameters in `benchmark.cu` are optimized for a Nvidia T2000 so depending on the GPU used, tweaking the parameters might give a considerable speedup.