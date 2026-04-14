#include "flip_kernels.h"
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <cstring>

// CUDA kernel to convert planar RGB to interleaved RGBA and write directly to
// CUDA surface (OpenGL texture) without modifying orientation.
__global__ void planarRGBToInterleavedRGBASurfaceKernel(
    const float* __restrict__ planarRGB,
    const float* __restrict__ alpha,
    cudaSurfaceObject_t outputSurface,
    int width,
    int height) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= width || y >= height) {
    return;
  }
  
  // Input planar format: [R0...RN, G0...GN, B0...BN]
  int planeSize = width * height;
  int srcIdx = y * width + x;
  
  // Read from planar RGB
  float4 rgba;
  rgba.x = planarRGB[0 * planeSize + srcIdx]; // R
  rgba.y = planarRGB[1 * planeSize + srcIdx]; // G
  rgba.z = planarRGB[2 * planeSize + srcIdx]; // B
  if (alpha) {
    float a = alpha[srcIdx];
    rgba.w = fminf(fmaxf(a, 0.0f), 1.0f);
  } else {
    rgba.w = 1.0f;
  }
  
  // Write directly to surface (OpenGL texture). Incoming image is Y-down,
  // so flip vertically to match OpenGL's Y-up convention.
  int destY = height - 1 - y;
  surf2Dwrite(rgba, outputSurface, x * sizeof(float4), destY);
}

// CUDA kernel to write depth data directly to CUDA surface without flipping
__global__ void depthToSurfaceKernel(
    const float* __restrict__ inputDepth,
    const float* __restrict__ inputAlpha,
    cudaSurfaceObject_t outputSurface,
    int width,
    int height,
    float nearPlane,
    float farPlane,
    float alphaCutoff) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= width || y >= height) {
    return;
  }
  
  int srcIdx = y * width + x;

  // Gaussian rasterizer now outputs expected linear depth (view space +Z
  // forward). If missing/invalid, push to far plane.
  float linearDepth = inputDepth[srcIdx];
  if (inputAlpha) {
    float alpha = inputAlpha[srcIdx];
    if (alpha <= alphaCutoff) {
      linearDepth = farPlane;
    }
  }
  if (linearDepth <= 0.0f)
    linearDepth = farPlane;
  // Convert linear view-space depth (positive Z forward) to OpenGL depth
  // Normalize to [0, 1] window-space depth
  const float eps = 1e-6f;
  float denom = fmaxf(linearDepth, eps);
  float ndcDepth =
      (farPlane + nearPlane) / (farPlane - nearPlane) -
      (2.0f * farPlane * nearPlane) / ((farPlane - nearPlane) * denom);
  float depth = ndcDepth * 0.5f + 0.5f;
  depth = fminf(fmaxf(depth, 0.0f), 1.0f);
  
  // Write directly to surface (OpenGL texture) with vertical flip to counter
  // the Y-down rasterizer outputs.
  int destY = height - 1 - y;
  surf2Dwrite(depth, outputSurface, x * sizeof(float), destY);
}

cudaSurfaceObject_t createSurfaceObject(cudaArray_t outputArray) {
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = outputArray;

  cudaSurfaceObject_t outputSurface = 0;
  cudaError_t err = cudaCreateSurfaceObject(&outputSurface, &resDesc);
  if (err != cudaSuccess) {
    return 0;
  }
  return outputSurface;
}

void destroySurfaceObject(cudaSurfaceObject_t surface) {
  if (surface != 0) {
    cudaDestroySurfaceObject(surface);
  }
}

// Host function to launch planar RGB to interleaved RGBA conversion kernel
void launchPlanarRGBToInterleavedRGBASurface(
    const float* d_planarRGB,
    const float* d_alpha,
    cudaSurfaceObject_t outputSurface,
    int width,
    int height,
    cudaStream_t stream) {
  if (outputSurface == 0) {
    return;
  }

  // Use 16x16 thread blocks for good occupancy
  dim3 blockDim(16, 16);
  dim3 gridDim(
      (width + blockDim.x - 1) / blockDim.x,
      (height + blockDim.y - 1) / blockDim.y);

  planarRGBToInterleavedRGBASurfaceKernel<<<gridDim, blockDim, 0, stream>>>(
      d_planarRGB,
      d_alpha,
      outputSurface,
      width,
      height);
}

// Host function to launch depth kernel
void launchDepthToSurface(
    const float* d_inputDepth,
    const float* d_inputAlpha,
    cudaSurfaceObject_t outputSurface,
    int width,
    int height,
    float nearPlane,
    float farPlane,
    float alphaCutoff,
    cudaStream_t stream) {
  if (outputSurface == 0) {
    return;
  }

  // Use 16x16 thread blocks for good occupancy
  dim3 blockDim(16, 16);
  dim3 gridDim(
      (width + blockDim.x - 1) / blockDim.x,
      (height + blockDim.y - 1) / blockDim.y);

  depthToSurfaceKernel<<<gridDim, blockDim, 0, stream>>>(
      d_inputDepth,
      d_inputAlpha,
      outputSurface,
      width,
      height,
      nearPlane,
      farPlane,
      alphaCutoff);
}
