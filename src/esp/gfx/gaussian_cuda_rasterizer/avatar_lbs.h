#pragma once

#include <cuda_runtime_api.h>

// Launch LBS kernel for Gaussian Avatar points.
// Joint and inverse bind matrices are expected to be row-major 4x4 matrices.
void launchAvatarLBS(const float* canonicalPositions,
                     const float* offsets,
                     const float* skinningWeights,
                     const float* jointMatrices,
                     const float* invBindMatrices,
                     const float* canonicalRotations,
                     int jointCount,
                     int pointCount,
                     float* outPositions,
                     float* outRotations,
                     cudaStream_t stream);
