#include <memory>
#include <random>
#include <iostream>
#include <chrono>

#include "../include/exercises/VectorAdditionKernel.cuh"

constexpr int N = 16e6;

using namespace exercises;
using namespace std;
using namespace std::chrono;

int main()
{
  // Allocate memory on the host
  float* a = (float*)malloc(sizeof(float) * N);
  float* b = (float*)malloc(sizeof(float) * N);
  float* c_Dev = (float*)malloc(sizeof(float) * N);
  float* c_Host = (float*)malloc(sizeof(float) * N);

  // Set a and b to random vectors.
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < N; i++)
  {
    a[i] = dis(gen);
    b[i] = dis(gen);
  }

  auto start = chrono::high_resolution_clock::now();

  // Find the host control result
  for (size_t i = 0; i < N; i++)
    c_Host[i] = a[i] + b[i];

  auto stop = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> elapsed = stop - start;

  cout << "Host vector addition took " << elapsed.count() << " ms" << std::endl;

  // Find the device result
  VectorAdder adder{};
  adder.Add(a, b, c_Dev, N);

  // Verify results
  try
  {
    for (size_t i = 0; i < N; i++)
    {
      if (c_Dev[i] != c_Host[i])
        throw;
    }

    cout << "The test passed!!! Vector of size " << N << " has successfully been added.";
  }
  catch (const exception& e)
  {
    cerr << "The host and device results are not the same...";
  }

  free(a);
  free(b);
  free(c_Dev);
  free(c_Host);
}
