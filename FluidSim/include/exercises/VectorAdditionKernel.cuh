#include "../IntellisenseFix.h"
#include <cuda_runtime.h>

namespace exercises
{
  constexpr int ThreadsPerBlock = 256;

  __global__ void VecAddKernel(const float* a, const float* b, float* c, int n);

  class VectorAdder
  {
  public:
    /// <summary>
    /// Adds vector a and b together on the GPU.
    /// </summary>
    /// <param name="a">A vector of size n on the CPU.</param>
    /// <param name="b">A vector of size n on the CPU.</param>
    /// <param name="c">The resulting vector on the CPU.</param>
    /// <param name="n">The size of the vectors.</param>
    void Add(const float* a, const float* b, float* c, int n);
  };
}
