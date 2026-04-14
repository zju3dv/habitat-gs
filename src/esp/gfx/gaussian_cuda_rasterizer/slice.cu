#include "slice.h"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
__global__ void sliceKernel(const float* basePositions,
                            const float* baseScales,
                            const float* baseRotations,
                            const float* baseOpacities,
                            const float* times,
                            const float* timeScales,
                            const float* motion,
                            int motionStride,
                            int numGaussians,
                            float time,
                            float* outPositions,
                            float* outScales,
                            float* outRotations,
                            float* outOpacities) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numGaussians) {
    return;
  }

  const float dt = time - times[idx];
  const float sigmaT = expf(timeScales[idx]);
  const float denom = sigmaT * sigmaT + 1e-8f;
  const float temporal = expf(-0.5f * dt * dt / denom);

  // Motion terms (velocity + optional higher orders)
  const int motionBase = idx * motionStride;
  const float vx = motion[motionBase + 0];
  const float vy = motion[motionBase + 1];
  const float vz = motion[motionBase + 2];
  const float ax = (motionStride >= 6) ? motion[motionBase + 3] : 0.0f;
  const float ay = (motionStride >= 6) ? motion[motionBase + 4] : 0.0f;
  const float az = (motionStride >= 6) ? motion[motionBase + 5] : 0.0f;
  const float jx = (motionStride >= 9) ? motion[motionBase + 6] : 0.0f;
  const float jy = (motionStride >= 9) ? motion[motionBase + 7] : 0.0f;
  const float jz = (motionStride >= 9) ? motion[motionBase + 8] : 0.0f;
  const float dt2 = dt * dt;
  const float dt3 = dt2 * dt;

  // Position with linear motion
  const int posOffset = idx * 3;
  outPositions[posOffset + 0] =
      basePositions[posOffset + 0] +
      vx * dt +
      0.5f * ax * dt2 +
      (1.0f / 6.0f) * jx * dt3;
  outPositions[posOffset + 1] =
      basePositions[posOffset + 1] +
      vy * dt +
      0.5f * ay * dt2 +
      (1.0f / 6.0f) * jy * dt3;
  outPositions[posOffset + 2] =
      basePositions[posOffset + 2] +
      vz * dt +
      0.5f * az * dt2 +
      (1.0f / 6.0f) * jz * dt3;

  // Scales and rotations are already activated/normalized in base buffers
  outScales[posOffset + 0] = baseScales[posOffset + 0];
  outScales[posOffset + 1] = baseScales[posOffset + 1];
  outScales[posOffset + 2] = baseScales[posOffset + 2];

  const int rotOffset = idx * 4;
  outRotations[rotOffset + 0] = baseRotations[rotOffset + 0];
  outRotations[rotOffset + 1] = baseRotations[rotOffset + 1];
  outRotations[rotOffset + 2] = baseRotations[rotOffset + 2];
  outRotations[rotOffset + 3] = baseRotations[rotOffset + 3];

  // Opacity is pre-sigmoid; apply temporal falloff
  outOpacities[idx] = baseOpacities[idx] * temporal;
}
}  // namespace

void launchSliceGaussians(const float* basePositions,
                          const float* baseScales,
                          const float* baseRotations,
                          const float* baseOpacities,
                          const float* times,
                          const float* timeScales,
                          const float* motion,
                          int motionStride,
                          int numGaussians,
                          float time,
                          float* outPositions,
                          float* outScales,
                          float* outRotations,
                          float* outOpacities,
                          cudaStream_t stream) {
  if (numGaussians <= 0) {
    return;
  }

  dim3 block(256);
  dim3 grid((numGaussians + block.x - 1) / block.x);
  sliceKernel<<<grid, block, 0, stream>>>(
      basePositions, baseScales, baseRotations, baseOpacities, times,
      timeScales, motion, motionStride, numGaussians, time, outPositions,
      outScales, outRotations, outOpacities);
}
