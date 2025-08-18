#include "exercises/VectorAdditionKernel.cuh"

#include <iostream>

using namespace std;

namespace exercises
{
  __global__ void VecAddKernel(const float* a, const float* b, float* c, int n)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < n)
    {
      c[tid] = a[tid] + b[tid];
      tid += blockDim.x * gridDim.x;
    }
  }

  void VectorAdder::Add(const float* a, const float* b, float* c, int n)
  {
    float *dev_a, *dev_b, *dev_c;

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc((void**)&dev_a, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc((void**)&dev_c, sizeof(float) * n));

    // Copy memory to device
    CUDA_CHECK(cudaMemcpy(dev_a, a, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, sizeof(float) * n, cudaMemcpyHostToDevice));
    
    // Setup record process time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch the kernel
    int blocks = (n + ThreadsPerBlock - 1) / ThreadsPerBlock;
    VecAddKernel<<<blocks, ThreadsPerBlock>>>(dev_a, dev_b, dev_c, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Find the process time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Kernel execution time: " << milliseconds << " ms" << endl;

    // Copy back to host and cleanup
    CUDA_CHECK(cudaMemcpy(c, dev_c, sizeof(float) * n, cudaMemcpyDeviceToHost));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "Finished adding vectors." << endl;
  }
}
