#include "preprocess_kernels.h"

#include <cuda_runtime.h>

__global__ void preprocessGaussiansKernel(
    float* opacities,
    float* scales,
    float* rotations,
    int numGaussians,
    bool normalizeQuaternion) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numGaussians) {
    return;
  }

  // Sigmoid opacity
  float op = opacities[idx];
  opacities[idx] = 1.0f / (1.0f + expf(-op));

  // Exp scales
  float* scalePtr = scales + idx * 3;
  scalePtr[0] = expf(scalePtr[0]);
  scalePtr[1] = expf(scalePtr[1]);
  scalePtr[2] = expf(scalePtr[2]);

  if (normalizeQuaternion) {
    // Normalize quaternion (w, x, y, z) in place
    float* rotPtr = rotations + idx * 4;
    float w = rotPtr[0];
    float x = rotPtr[1];
    float y = rotPtr[2];
    float z = rotPtr[3];
    float norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-8f);
    rotPtr[0] = w * norm;
    rotPtr[1] = x * norm;
    rotPtr[2] = y * norm;
    rotPtr[3] = z * norm;
  }
}

void launchPreprocessGaussians(float* d_opacities,
                               float* d_scales,
                               float* d_rotations,
                               int numGaussians,
                               bool normalizeQuaternion,
                               cudaStream_t stream) {
  if (numGaussians <= 0) {
    return;
  }
  dim3 block(256);
  dim3 grid((numGaussians + block.x - 1) / block.x);
  preprocessGaussiansKernel<<<grid, block, 0, stream>>>(
      d_opacities, d_scales, d_rotations, numGaussians, normalizeQuaternion);
}
