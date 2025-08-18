#pragma once

#ifdef __INTELLISENSE__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__
#define __restrict__
#define __syncthreads()

// Dummy definitions to satisfy IntelliSense
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int X=0, unsigned int Y=0, unsigned int Z=0) noexcept : x(X), y(Y), z(Z) {}
};

static constexpr dim3 blockIdx{0,0,0};
static constexpr dim3 threadIdx{0,0,0};
static constexpr dim3 blockDim{0,0,0};
static constexpr dim3 gridDim{0,0,0};

#endif

// Some useful defines
#define CUDA_CHECK(err) \
  if (err != cudaSuccess) \
  { \
    cerr << "CUDA error: " << cudaGetErrorString(err) << endl; \
    exit(EXIT_FAILURE); \
  }
