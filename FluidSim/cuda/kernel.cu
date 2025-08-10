#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "IntellisenseFix.h"

extern "C" __global__ void fillGradient(uchar4* devPtr, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    unsigned char r = (unsigned char)((x + time * 100)) % 256;
    unsigned char g = (unsigned char)((y + time * 50)) % 256;
    unsigned char b = 128;
    devPtr[idx] = make_uchar4(r, g, b, 255);
}


extern "C" void runFillGradientKernel(uchar4* devPtr, int width, int height, float time) {
    dim3 blockSize(16,16);
    dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);

    fillGradient<<<gridSize, blockSize>>>(devPtr, width, height, time);
}

