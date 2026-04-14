#ifndef FLIP_KERNELS_H_
#define FLIP_KERNELS_H_

#include <cuda_runtime.h>

/**
 * @brief Convert planar RGB to interleaved RGBA format and write directly to a
 *        CUDA array (OpenGL texture) without modifying image orientation.
 *
 * Input planar format: [R0...RN, G0...GN, B0...BN]
 * Output interleaved format written to CUDA array: [R0,G0,B0,A0, R1,G1,B1,A1, ...]
 *
 * @param d_planarRGB Input planar RGB data (device pointer)
 * @param d_alpha Optional alpha data (device pointer, can be nullptr for A=1)
 * @param outputArray Output CUDA array (mapped from OpenGL texture)
 * @param width Image width
 * @param height Image height
 * @param stream CUDA stream (default 0)
 */
cudaSurfaceObject_t createSurfaceObject(cudaArray_t outputArray);

void destroySurfaceObject(cudaSurfaceObject_t surface);

void launchPlanarRGBToInterleavedRGBASurface(
    const float* d_planarRGB,
    const float* d_alpha,
    cudaSurfaceObject_t outputSurface,
    int width,
    int height,
    cudaStream_t stream = 0);

/**
 * @brief Write depth data directly to a CUDA array (OpenGL texture) without
 *        altering orientation.
 *
 * @param d_inputDepth Input depth data (device pointer)
 * @param d_inputAlpha Optional alpha data (device pointer, can be nullptr)
 * @param outputArray Output CUDA array (mapped from OpenGL texture)
 * @param width Image width
 * @param height Image height
 * @param alphaCutoff Alpha threshold below which depth is pushed to far plane
 * @param stream CUDA stream (default 0)
 */
void launchDepthToSurface(
    const float* d_inputDepth,
    const float* d_inputAlpha,
    cudaSurfaceObject_t outputSurface,
    int width,
    int height,
    float nearPlane,
    float farPlane,
    float alphaCutoff,
    cudaStream_t stream = 0);

#endif  // FLIP_KERNELS_H_
