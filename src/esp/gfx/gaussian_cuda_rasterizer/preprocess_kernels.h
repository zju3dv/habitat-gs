#ifndef PREPROCESS_KERNELS_H_
#define PREPROCESS_KERNELS_H_

#include <cuda_runtime.h>

/**
 * @brief One-time preprocessing for Gaussian parameters on GPU.
 *
 * Applies sigmoid to opacities, exp to scales, and optionally normalizes quaternions
 * (w, x, y, z) in-place.
 */
void launchPreprocessGaussians(float* d_opacities,
                               float* d_scales,
                               float* d_rotations,
                               int numGaussians,
                               bool normalizeQuaternion = true,
                               cudaStream_t stream = 0);

#endif  // PREPROCESS_KERNELS_H_
