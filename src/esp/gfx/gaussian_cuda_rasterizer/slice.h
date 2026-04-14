#pragma once

#include <cuda_runtime_api.h>

// Forward declaration for CUDA slicing kernel. Converts 4D Gaussian parameters
// into per-frame 3D attributes used by the rasterizer.
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
                          cudaStream_t stream);
