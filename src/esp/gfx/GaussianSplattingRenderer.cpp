#include "GaussianSplattingRenderer.h"

#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Debug.h>
#include <Magnum/GL/Context.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

#include "esp/core/Check.h"
#include "esp/gfx/configure.h"

#ifdef ESP_BUILD_WITH_CUDA
#include "gaussian_cuda_rasterizer/rasterizer.h"
#include "gaussian_cuda_rasterizer/flip_kernels.h"
#include "gaussian_cuda_rasterizer/preprocess_kernels.h"
#include "gaussian_cuda_rasterizer/slice.h"
#include "gaussian_cuda_rasterizer/avatar_lbs.h"
#include "esp/gfx_batch/cuda_helpers/HelperCuda.h"
#endif

namespace Cr = Corrade;
namespace Mn = Magnum;

namespace esp {
namespace gfx {

void beginAvatarBatchRenderPass(
    const std::vector<GaussianSplattingRenderer*>& renderers);
void endAvatarBatchRenderPass();

namespace {
int deduceShDegree(int shRestCount) {
  if (shRestCount <= 0) {
    return 0;
  }
  double coeffs = static_cast<double>(shRestCount) / 3.0 + 1.0;
  return static_cast<int>(std::max(0.0, std::round(std::sqrt(coeffs) - 1.0)));
}

struct AvatarBatchContext {
  bool active = false;
  std::vector<GaussianSplattingRenderer*> renderers;
  GaussianSplattingRenderer* primary = nullptr;
};

struct AvatarBatchCache {
  bool valid = false;
  std::vector<GaussianSplattingRenderer*> rendererSignature;
  std::vector<size_t> updateSignature;
  size_t totalGaussians = 0;
  int shDim = 0;
  int shDegree = 0;
#ifdef ESP_BUILD_WITH_CUDA
  float* d_means3D = nullptr;
  float* d_shs = nullptr;
  float* d_opacities = nullptr;
  float* d_scales = nullptr;
  float* d_rotations = nullptr;
  size_t meansCapacity = 0;
  size_t shsCapacity = 0;
  size_t opacitiesCapacity = 0;
  size_t scalesCapacity = 0;
  size_t rotationsCapacity = 0;
#endif
};

AvatarBatchContext gAvatarBatchContext;
AvatarBatchCache gAvatarBatchCache;

#ifdef ESP_BUILD_WITH_CUDA
void ensureAvatarBatchBufferCapacity(float*& buffer,
                                     size_t& capacity,
                                     size_t requiredCount,
                                     const char* debugName) {
  if (capacity >= requiredCount && buffer != nullptr) {
    return;
  }
  if (buffer != nullptr) {
    cudaFree(buffer);
    buffer = nullptr;
    capacity = 0;
  }
  if (requiredCount == 0) {
    return;
  }
  cudaError_t err = cudaMalloc(&buffer, requiredCount * sizeof(float));
  if (err != cudaSuccess) {
    ESP_ERROR() << "Failed to allocate avatar batch buffer" << debugName
                << "count=" << requiredCount << "error="
                << cudaGetErrorString(err);
    checkCudaErrors(err);
  }
  capacity = requiredCount;
}

void freeAvatarBatchCacheDeviceBuffers() {
  if (gAvatarBatchCache.d_means3D) {
    cudaFree(gAvatarBatchCache.d_means3D);
  }
  if (gAvatarBatchCache.d_shs) {
    cudaFree(gAvatarBatchCache.d_shs);
  }
  if (gAvatarBatchCache.d_opacities) {
    cudaFree(gAvatarBatchCache.d_opacities);
  }
  if (gAvatarBatchCache.d_scales) {
    cudaFree(gAvatarBatchCache.d_scales);
  }
  if (gAvatarBatchCache.d_rotations) {
    cudaFree(gAvatarBatchCache.d_rotations);
  }
  gAvatarBatchCache.d_means3D = nullptr;
  gAvatarBatchCache.d_shs = nullptr;
  gAvatarBatchCache.d_opacities = nullptr;
  gAvatarBatchCache.d_scales = nullptr;
  gAvatarBatchCache.d_rotations = nullptr;
  gAvatarBatchCache.meansCapacity = 0;
  gAvatarBatchCache.shsCapacity = 0;
  gAvatarBatchCache.opacitiesCapacity = 0;
  gAvatarBatchCache.scalesCapacity = 0;
  gAvatarBatchCache.rotationsCapacity = 0;
  gAvatarBatchCache.valid = false;
  gAvatarBatchCache.rendererSignature.clear();
  gAvatarBatchCache.updateSignature.clear();
  gAvatarBatchCache.totalGaussians = 0;
  gAvatarBatchCache.shDim = 0;
  gAvatarBatchCache.shDegree = 0;
}

struct AvatarBatchCacheFinalizer {
  ~AvatarBatchCacheFinalizer() { freeAvatarBatchCacheDeviceBuffers(); }
} gAvatarBatchCacheFinalizer;
#endif
}  // namespace

void beginAvatarBatchRenderPass(
    const std::vector<GaussianSplattingRenderer*>& renderers) {
#ifdef ESP_BUILD_WITH_CUDA
  gAvatarBatchContext.active = !renderers.empty();
  gAvatarBatchContext.renderers = renderers;
  gAvatarBatchContext.primary =
      renderers.empty() ? nullptr : gAvatarBatchContext.renderers.front();
#else
  static_cast<void>(renderers);
#endif
}

void endAvatarBatchRenderPass() {
#ifdef ESP_BUILD_WITH_CUDA
  gAvatarBatchContext.active = false;
  gAvatarBatchContext.renderers.clear();
  gAvatarBatchContext.primary = nullptr;
#endif
}

// ============================================================================
// CudaGLInterop implementation
// ============================================================================

CudaGLInterop::~CudaGLInterop() {
  release();
}

void CudaGLInterop::release() {
#ifdef ESP_BUILD_WITH_CUDA
  if (!registered_) {
    mapped_ = false;
    cudaResource_ = nullptr;
    registeredTextureId_ = 0;
    return;
  }

  if (mapped_) {
    cudaError_t unmapErr = cudaGraphicsUnmapResources(1, &cudaResource_, 0);
    if (unmapErr != cudaSuccess) {
      ESP_WARNING() << "Failed to unmap CUDA-GL resource during cleanup:"
                    << cudaGetErrorString(unmapErr);
    }
    mapped_ = false;
  }

  const bool hasGlContext = Mn::GL::Context::hasCurrent();
  const bool textureStillExists =
      hasGlContext && registeredTextureId_ != 0 &&
      glIsTexture(registeredTextureId_);

  if (!hasGlContext) {
    ESP_WARNING()
        << "Skipping CUDA-GL unregister because no current GL context exists";
  } else if (registeredTextureId_ != 0 && !textureStillExists) {
    ESP_WARNING() << "Skipping CUDA-GL unregister for destroyed texture"
                  << registeredTextureId_;
  } else {
    cudaError_t unregisterErr = cudaGraphicsUnregisterResource(cudaResource_);
    if (unregisterErr != cudaSuccess) {
      ESP_WARNING() << "Failed to unregister CUDA-GL resource during cleanup:"
                    << cudaGetErrorString(unregisterErr);
    }
  }

  registered_ = false;
  registeredTextureId_ = 0;
  cudaResource_ = nullptr;
#endif
}

void CudaGLInterop::registerTexture(Mn::GL::Texture2D& texture,
                                    unsigned int glTextureId) {
#ifdef ESP_BUILD_WITH_CUDA
  // Re-register if texture id has changed
  if (registered_) {
    if (registeredTextureId_ == glTextureId) {
      return;
    }
    release();
  }

  GLenum staleError = GL_NO_ERROR;
  while ((staleError = glGetError()) != GL_NO_ERROR) {
    ESP_DEBUG() << "Clearing pre-existing OpenGL error before CUDA interop:"
                << staleError;
  }
  
  // Bind the texture to make it current - this is CRITICAL for CUDA registration
  texture.bind(0);
  
  // Check for GL errors
  GLenum glError = glGetError();
  if (glError != GL_NO_ERROR) {
    ESP_ERROR() << "OpenGL error before CUDA registration:" << glError;
  }
  
  // Verify texture is valid
  GLboolean isTexture = glIsTexture(glTextureId);
  ESP_DEBUG() << "glIsTexture():" << (isTexture ? "YES" : "NO");
  if (!isTexture) {
    ESP_ERROR() << "Invalid GL texture ID:" << glTextureId;
  }
  
  // Check CUDA device
  int cudaDevice;
  cudaError_t err = cudaGetDevice(&cudaDevice);
  ESP_DEBUG() << "CUDA Device:" << cudaDevice << "Error:" << cudaGetErrorString(err);
  
  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cudaDevice);
  ESP_DEBUG() << "CUDA Device Name:" << prop.name;
  ESP_DEBUG() << "Compute Capability:" << prop.major << "." << prop.minor;
  
  // Ensure all GL operations are complete
  glFinish();
  
  // Ensure OpenGL context is current
  ESP_DEBUG() << "Attempting cudaGraphicsGLRegisterImage...";
  
  err = cudaGraphicsGLRegisterImage(
      &cudaResource_, 
      glTextureId,
      GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsWriteDiscard);
  
  if (err != cudaSuccess) {
    ESP_ERROR() << "cudaGraphicsGLRegisterImage FAILED!";
    ESP_ERROR() << "Error code:" << err << "(" << cudaGetErrorString(err) << ")";
    ESP_ERROR() << "GL Texture ID:" << glTextureId;
    checkCudaErrors(err);  // This will throw
  }
  
  registered_ = true;
  registeredTextureId_ = glTextureId;
  ESP_DEBUG() << "Successfully registered GL texture with CUDA";
#else
  ESP_CHECK(false, "CudaGLInterop requires CUDA support");
#endif
}

void CudaGLInterop::mapResources() {
#ifdef ESP_BUILD_WITH_CUDA
  ESP_CHECK(registered_ && !mapped_,
            "CudaGLInterop::mapResources(): Invalid state");
  
  checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource_, 0));
  mapped_ = true;
#endif
}

void CudaGLInterop::unmapResources() {
#ifdef ESP_BUILD_WITH_CUDA
  ESP_CHECK(mapped_, "CudaGLInterop::unmapResources(): Not mapped");
  
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResource_, 0));
  mapped_ = false;
#endif
}

#ifdef ESP_BUILD_WITH_CUDA
cudaArray_t CudaGLInterop::getMappedArray() {
  ESP_CHECK(mapped_, "CudaGLInterop::getMappedArray(): Resources not mapped");
  
  cudaArray_t array = nullptr;
  checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, cudaResource_, 0, 0));
  return array;
}
#endif

// ============================================================================
// GaussianSplattingRenderer implementation
// ============================================================================

GaussianSplattingRenderer::GaussianSplattingRenderer() {
#ifdef ESP_BUILD_WITH_CUDA
  // Set up CUDA to work with current OpenGL context
  int deviceCount = 0;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  
  if (deviceCount == 0) {
    ESP_ERROR() << "No CUDA devices found!";
    return;
  }
  
  // Find CUDA device compatible with OpenGL
  int cudaDeviceId = 0;
  unsigned int devicesCount = 0;
  cudaError_t err = cudaGLGetDevices(&devicesCount, &cudaDeviceId, 1, 
                                      cudaGLDeviceListAll);
  
  if (err == cudaSuccess && devicesCount > 0) {
    checkCudaErrors(cudaSetDevice(cudaDeviceId));
  } else {
    ESP_WARNING() << "cudaGLGetDevices failed or no GL-compatible devices found";
    ESP_WARNING() << "Error:" << cudaGetErrorString(err);
    checkCudaErrors(cudaSetDevice(0));
  }
  
  // Verify device was set
  int currentDevice;
  checkCudaErrors(cudaGetDevice(&currentDevice));
  ESP_DEBUG() << "Current CUDA device:" << currentDevice;
  
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, currentDevice));
  ESP_DEBUG() << "Device name:" << prop.name;
  ESP_DEBUG() << "Compute capability:" << prop.major << "." << prop.minor;
  
  colorInterop_ = std::make_unique<CudaGLInterop>();
  depthInterop_ = std::make_unique<CudaGLInterop>();
  
  ESP_DEBUG() << "GaussianSplattingRenderer initialized";
#endif
}

GaussianSplattingRenderer::~GaussianSplattingRenderer() {
  releaseInteropResources();
  freeGPUBuffers();
}

void GaussianSplattingRenderer::releaseInteropResources() {
#ifdef ESP_BUILD_WITH_CUDA
  if (colorInterop_) {
    colorInterop_->release();
  }
  if (depthInterop_) {
    depthInterop_->release();
  }
#endif
}

void GaussianSplattingRenderer::initialize(
    const assets::GaussianSplattingData& gaussianData,
    bool normalizeQuaternion) {
#ifndef ESP_BUILD_WITH_CUDA
  ESP_CHECK(false, 
            "GaussianSplattingRenderer::initialize(): CUDA support required");
  return;
#else
  
  // Free existing buffers if any
  freeGPUBuffers();
  avatarMode_ = false;
  avatarJointCount_ = 0;

  foregroundCount_ =
      static_cast<int>(gaussianData.getDynamicGaussianCount());
  backgroundCount_ =
      static_cast<int>(gaussianData.getStaticGaussianCount());
  numGaussians_ = foregroundCount_ + backgroundCount_;
  is4D_ = foregroundCount_ > 0;
  motionStride_ =
      std::max(3, std::min(9, gaussianData.getMotionStride()));
  ESP_CHECK(numGaussians_ > 0,
            "GaussianSplattingRenderer::initialize(): No Gaussians to render");
  
  // Get SH dimensions
  size_t shRestCount = gaussianData.getSHRestCount();
  shDim_ = 3 + shRestCount;  // 3 for DC + rest
  shDegree_ = deduceShDegree(static_cast<int>(shRestCount));
  // Allocate GPU buffers
  prepareGPUBuffers();

  // Stream upload to GPU in chunks to avoid large CPU-side duplicates
  const size_t chunkSize = 100000;  // tuneable to balance memory vs. copy overhead
  std::vector<float> meansChunk;
  std::vector<float> shsChunk;
  std::vector<float> opacitiesChunk;
  std::vector<float> scalesChunk;
  std::vector<float> rotationsChunk;
  std::vector<float> timesChunk;
  std::vector<float> timeScalesChunk;
  std::vector<float> motionsChunk;

  meansChunk.reserve(chunkSize * 3);
  shsChunk.reserve(chunkSize * shDim_);
  opacitiesChunk.reserve(chunkSize);
  scalesChunk.reserve(chunkSize * 3);
  rotationsChunk.reserve(chunkSize * 4);
  timesChunk.reserve(chunkSize);
  timeScalesChunk.reserve(chunkSize);
  motionsChunk.reserve(chunkSize * motionStride_);

  auto uploadDynamicSet =
      [&](const std::vector<assets::GaussianSplat4D>& gaussians) {
        for (size_t start = 0; start < gaussians.size(); start += chunkSize) {
          size_t count =
              std::min(chunkSize, gaussians.size() - start);
          meansChunk.assign(count * 3, 0.0f);
          shsChunk.assign(count * shDim_, 0.0f);
          opacitiesChunk.assign(count, 0.0f);
          scalesChunk.assign(count * 3, 0.0f);
          rotationsChunk.assign(count * 4, 0.0f);
          timesChunk.assign(count, 0.0f);
          timeScalesChunk.assign(count, 0.0f);
          motionsChunk.assign(count * motionStride_, 0.0f);

          for (size_t i = 0; i < count; ++i) {
            const auto& g = gaussians[start + i];
            meansChunk[i * 3 + 0] = g.position.x();
            meansChunk[i * 3 + 1] = g.position.y();
            meansChunk[i * 3 + 2] = g.position.z();

            shsChunk[i * shDim_ + 0] = g.f_dc.x();
            shsChunk[i * shDim_ + 1] = g.f_dc.y();
            shsChunk[i * shDim_ + 2] = g.f_dc.z();
            size_t numCoeffs = shRestCount / 3;
              for (size_t m = 0; m < numCoeffs; ++m) {
                for (size_t c = 0; c < 3; ++c) {
                  size_t srcIdx = c * numCoeffs + m;
                  size_t dstIdx = i * shDim_ + 3 + (m * 3 + c);
                  if (srcIdx < g.f_rest.size()) {
                    shsChunk[dstIdx] = g.f_rest[srcIdx];
                  }
                }
              }

            opacitiesChunk[i] = g.opacity;
            scalesChunk[i * 3 + 0] = g.scale.x();
            scalesChunk[i * 3 + 1] = g.scale.y();
            scalesChunk[i * 3 + 2] = g.scale.z();
            rotationsChunk[i * 4 + 0] = g.rotation.scalar();
            rotationsChunk[i * 4 + 1] = g.rotation.vector().x();
            rotationsChunk[i * 4 + 2] = g.rotation.vector().y();
            rotationsChunk[i * 4 + 3] = g.rotation.vector().z();

            timesChunk[i] = g.time;
            timeScalesChunk[i] = g.timeScale;
            motionsChunk[i * motionStride_ + 0] = g.motion.x();
            motionsChunk[i * motionStride_ + 1] = g.motion.y();
            motionsChunk[i * motionStride_ + 2] = g.motion.z();
            if (motionStride_ >= 6) {
              motionsChunk[i * motionStride_ + 3] = g.motionAccel.x();
              motionsChunk[i * motionStride_ + 4] = g.motionAccel.y();
              motionsChunk[i * motionStride_ + 5] = g.motionAccel.z();
            }
            if (motionStride_ >= 9) {
              motionsChunk[i * motionStride_ + 6] = g.motionJerk.x();
              motionsChunk[i * motionStride_ + 7] = g.motionJerk.y();
              motionsChunk[i * motionStride_ + 8] = g.motionJerk.z();
            }
          }

          size_t floatBytes = sizeof(float);
          checkCudaErrors(cudaMemcpy(d_baseMeans3D_ + start * 3,
                                     meansChunk.data(),
                                     count * 3 * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_shs_ + start * shDim_,
                                     shsChunk.data(),
                                     count * shDim_ * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_baseOpacities_ + start,
                                     opacitiesChunk.data(),
                                     count * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_baseScales_ + start * 3,
                                     scalesChunk.data(),
                                     count * 3 * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_baseRotations_ + start * 4,
                                     rotationsChunk.data(),
                                     count * 4 * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_times_ + start,
                                     timesChunk.data(),
                                     count * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_timeScales_ + start,
                                     timeScalesChunk.data(),
                                     count * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_motion_ + start * motionStride_,
                                     motionsChunk.data(),
                                     count * motionStride_ * floatBytes,
                                     cudaMemcpyHostToDevice));
        }
      };

  auto uploadStaticSet =
      [&](const std::vector<assets::GaussianSplat>& gaussians,
          size_t combinedOffset) {
        for (size_t start = 0; start < gaussians.size(); start += chunkSize) {
          size_t count =
              std::min(chunkSize, gaussians.size() - start);
          meansChunk.assign(count * 3, 0.0f);
          shsChunk.assign(count * shDim_, 0.0f);
          opacitiesChunk.assign(count, 0.0f);
          scalesChunk.assign(count * 3, 0.0f);
          rotationsChunk.assign(count * 4, 0.0f);

          for (size_t i = 0; i < count; ++i) {
            const auto& g = gaussians[start + i];
            meansChunk[i * 3 + 0] = g.position.x();
            meansChunk[i * 3 + 1] = g.position.y();
            meansChunk[i * 3 + 2] = g.position.z();

            shsChunk[i * shDim_ + 0] = g.f_dc.x();
            shsChunk[i * shDim_ + 1] = g.f_dc.y();
            shsChunk[i * shDim_ + 2] = g.f_dc.z();
            // f_rest is channel-major; interleave to match vec3 SH layout.
            const size_t numCoeffs = shRestCount / 3;
            for (size_t m = 0; m < numCoeffs; ++m) {
              for (size_t c = 0; c < 3; ++c) {
                const size_t srcIdx = c * numCoeffs + m;
                const size_t dstIdx = i * shDim_ + 3 + (m * 3 + c);
                if (srcIdx < g.f_rest.size()) {
                  shsChunk[dstIdx] = g.f_rest[srcIdx];
                }
              }
            }

            opacitiesChunk[i] = g.opacity;
            scalesChunk[i * 3 + 0] = g.scale.x();
            scalesChunk[i * 3 + 1] = g.scale.y();
            scalesChunk[i * 3 + 2] = g.scale.z();
            rotationsChunk[i * 4 + 0] = g.rotation.scalar();
            rotationsChunk[i * 4 + 1] = g.rotation.vector().x();
            rotationsChunk[i * 4 + 2] = g.rotation.vector().y();
            rotationsChunk[i * 4 + 3] = g.rotation.vector().z();
          }

          size_t floatBytes = sizeof(float);
          const size_t globalStart = combinedOffset + start;
          checkCudaErrors(cudaMemcpy(d_means3D_ + globalStart * 3,
                                     meansChunk.data(),
                                     count * 3 * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_shs_ + globalStart * shDim_,
                                     shsChunk.data(),
                                     count * shDim_ * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_opacities_ + globalStart,
                                     opacitiesChunk.data(),
                                     count * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_scales_ + globalStart * 3,
                                     scalesChunk.data(),
                                     count * 3 * floatBytes,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(d_rotations_ + globalStart * 4,
                                     rotationsChunk.data(),
                                     count * 4 * floatBytes,
                                     cudaMemcpyHostToDevice));
        }
      };

  uploadDynamicSet(gaussianData.getDynamicGaussians());
  uploadStaticSet(gaussianData.getStaticGaussians(),
                  static_cast<size_t>(foregroundCount_));

  // One-time GPU preprocess: sigmoid opacity, exp scale, optionally normalize quaternions
  if (foregroundCount_ > 0) {
    launchPreprocessGaussians(d_baseOpacities_, d_baseScales_,
                              d_baseRotations_, foregroundCount_, normalizeQuaternion, 0);
  }
  if (backgroundCount_ > 0) {
    launchPreprocessGaussians(d_opacities_ + foregroundCount_,
                              d_scales_ + foregroundCount_ * 3,
                              d_rotations_ + foregroundCount_ * 4,
                              backgroundCount_, normalizeQuaternion, 0);
  }
  checkCudaErrors(cudaGetLastError());
  
  initialized_ = true;
  
  ESP_DEBUG() << "GaussianSplattingRenderer initialized with" << numGaussians_
              << "Gaussians (foreground" << foregroundCount_ << ", background"
              << backgroundCount_ << ")";
#endif
}

void GaussianSplattingRenderer::initializeAvatar(
    const assets::GaussianAvatarData& avatarData) {
#ifndef ESP_BUILD_WITH_CUDA
  ESP_CHECK(false,
            "GaussianSplattingRenderer::initializeAvatar(): CUDA support required");
  return;
#else
  freeGPUBuffers();

  ESP_CHECK(avatarData.hasData(),
            "GaussianSplattingRenderer::initializeAvatar(): invalid avatar data");

  avatarMode_ = true;
  avatarJointCount_ = avatarData.getJointCount();
  numGaussians_ = avatarData.getPointCount();
  foregroundCount_ = 0;
  backgroundCount_ = numGaussians_;
  is4D_ = false;
  motionStride_ = 3;
  shDim_ = avatarData.getShDim();
  shDegree_ = avatarData.getShDegree();

  ESP_CHECK(numGaussians_ > 0 && avatarJointCount_ > 0,
            "GaussianSplattingRenderer::initializeAvatar(): empty avatar");

  prepareGPUBuffers();

  // Allocate avatar-specific buffers
  checkCudaErrors(cudaMalloc(&d_avatarCanonical_,
                             numGaussians_ * 3 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_avatarWeights_,
                             numGaussians_ * avatarJointCount_ *
                                 sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_avatarOffsets_,
                             numGaussians_ * 3 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_avatarCanonicalRotations_,
                             numGaussians_ * 4 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_avatarJointMatrices_,
                             avatarJointCount_ * 16 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_avatarInvBind_,
                             avatarJointCount_ * 16 * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_avatarCanonical_,
                             avatarData.getMeans().data(),
                             numGaussians_ * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_avatarWeights_,
                             avatarData.getSkinningWeights().data(),
                             numGaussians_ * avatarJointCount_ *
                                 sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_avatarInvBind_,
                             avatarData.getInvBindMatrices().data(),
                             avatarJointCount_ * 16 * sizeof(float),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemset(d_avatarOffsets_, 0,
                             numGaussians_ * 3 * sizeof(float)));

  // Initialize positions to canonical for the first frame.
  checkCudaErrors(cudaMemcpy(d_means3D_, d_avatarCanonical_,
                             numGaussians_ * 3 * sizeof(float),
                             cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemcpy(d_opacities_, avatarData.getOpacities().data(),
                             numGaussians_ * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_rotations_, avatarData.getRotations().data(),
                             numGaussians_ * 4 * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_avatarCanonicalRotations_,
                 avatarData.getRotations().data(),
                 numGaussians_ * 4 * sizeof(float),
                 cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_shs_, avatarData.getShs().data(),
                             numGaussians_ * shDim_ * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_scales_, avatarData.getScales().data(),
                             numGaussians_ * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));

  initialized_ = true;
  ESP_DEBUG() << "GaussianSplattingRenderer initialized avatar with"
              << numGaussians_ << "points and" << avatarJointCount_
              << "joints.";
#endif
}

void GaussianSplattingRenderer::prepareGPUBuffers() {
#ifdef ESP_BUILD_WITH_CUDA
  // Allocate GPU memory
  checkCudaErrors(cudaMalloc(&d_means3D_, numGaussians_ * 3 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_shs_, numGaussians_ * shDim_ * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_opacities_, numGaussians_ * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_scales_, numGaussians_ * 3 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_rotations_, numGaussians_ * 4 * sizeof(float)));

  if (foregroundCount_ > 0) {
    checkCudaErrors(
        cudaMalloc(&d_baseMeans3D_, foregroundCount_ * 3 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_baseOpacities_,
                               foregroundCount_ * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&d_baseScales_, foregroundCount_ * 3 * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&d_baseRotations_, foregroundCount_ * 4 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_times_, foregroundCount_ * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&d_timeScales_, foregroundCount_ * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&d_motion_,
                   foregroundCount_ * motionStride_ * sizeof(float)));
  }
#endif
}

void GaussianSplattingRenderer::freeGPUBuffers() {
#ifdef ESP_BUILD_WITH_CUDA
  if (d_means3D_) cudaFree(d_means3D_);
  if (d_shs_) cudaFree(d_shs_);
  if (d_opacities_) cudaFree(d_opacities_);
  if (d_scales_) cudaFree(d_scales_);
  if (d_rotations_) cudaFree(d_rotations_);
  if (d_avatarCanonical_) cudaFree(d_avatarCanonical_);
  if (d_avatarWeights_) cudaFree(d_avatarWeights_);
  if (d_avatarOffsets_) cudaFree(d_avatarOffsets_);
  if (d_avatarCanonicalRotations_) cudaFree(d_avatarCanonicalRotations_);
  if (d_avatarJointMatrices_) cudaFree(d_avatarJointMatrices_);
  if (d_avatarInvBind_) cudaFree(d_avatarInvBind_);
  if (d_baseMeans3D_) cudaFree(d_baseMeans3D_);
  if (d_baseOpacities_) cudaFree(d_baseOpacities_);
  if (d_baseScales_) cudaFree(d_baseScales_);
  if (d_baseRotations_) cudaFree(d_baseRotations_);
  if (d_times_) cudaFree(d_times_);
  if (d_timeScales_) cudaFree(d_timeScales_);
  if (d_motion_) cudaFree(d_motion_);
  if (d_outputColor_) cudaFree(d_outputColor_);
  if (d_outputDepth_) cudaFree(d_outputDepth_);
  if (d_outputAlpha_) cudaFree(d_outputAlpha_);
  if (d_debugRadii_) cudaFree(d_debugRadii_);
  if (d_viewMatrix_) cudaFree(d_viewMatrix_);
  if (d_projMatrix_) cudaFree(d_projMatrix_);
  if (d_camPos_) cudaFree(d_camPos_);
  if (d_background_) cudaFree(d_background_);
  if (geometryScratchBuffer_) cudaFree(geometryScratchBuffer_);
  if (binningScratchBuffer_) cudaFree(binningScratchBuffer_);
  if (imageScratchBuffer_) cudaFree(imageScratchBuffer_);
  
  d_means3D_ = nullptr;
  d_shs_ = nullptr;
  d_opacities_ = nullptr;
  d_scales_ = nullptr;
  d_rotations_ = nullptr;
  d_avatarCanonical_ = nullptr;
  d_avatarWeights_ = nullptr;
  d_avatarOffsets_ = nullptr;
  d_avatarCanonicalRotations_ = nullptr;
  d_avatarJointMatrices_ = nullptr;
  d_avatarInvBind_ = nullptr;
  d_baseMeans3D_ = nullptr;
  d_baseOpacities_ = nullptr;
  d_baseScales_ = nullptr;
  d_baseRotations_ = nullptr;
  d_times_ = nullptr;
  d_timeScales_ = nullptr;
  d_motion_ = nullptr;
  d_outputColor_ = nullptr;
  d_outputDepth_ = nullptr;
  d_outputAlpha_ = nullptr;
  d_debugRadii_ = nullptr;
  d_viewMatrix_ = nullptr;
  d_projMatrix_ = nullptr;
  d_camPos_ = nullptr;
  d_background_ = nullptr;
  geometryScratchBuffer_ = nullptr;
  binningScratchBuffer_ = nullptr;
  imageScratchBuffer_ = nullptr;
  geometryScratchCapacity_ = 0;
  binningScratchCapacity_ = 0;
  imageScratchCapacity_ = 0;
  lastWidth_ = 0;
  lastHeight_ = 0;
  foregroundCount_ = 0;
  backgroundCount_ = 0;
#endif

#ifdef ESP_BUILD_WITH_CUDA
  if (avatarMode_ && gAvatarBatchCache.valid) {
    gAvatarBatchCache.valid = false;
    gAvatarBatchCache.rendererSignature.clear();
    gAvatarBatchCache.updateSignature.clear();
  }
  if (gAvatarBatchContext.active && gAvatarBatchContext.primary == this) {
    gAvatarBatchContext.active = false;
    gAvatarBatchContext.renderers.clear();
    gAvatarBatchContext.primary = nullptr;
  }
#endif
  
  is4D_ = false;
  motionStride_ = 3;
  numGaussians_ = 0;
  shDim_ = 16;
  shDegree_ = 3;
  avatarJointCount_ = 0;
  avatarMode_ = false;
  avatarUpdateCalls_ = 0;
  avatarRenderCalls_ = 0;
  avatarRendersReusingPose_ = 0;
  avatarLastRenderedUpdateCall_ = 0;
  initialized_ = false;
}

void GaussianSplattingRenderer::matrixToArray(const Mn::Matrix4& matrix,
                                              float* array) {
  // Both Magnum and 3DGS CUDA rasterizer use column-major matrices
  // Magnum: matrix[col][row]
  // CUDA: matrix[col * 4 + row]
  // So we copy directly without transposing
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      array[col * 4 + row] = matrix[col][row];
    }
  }
}

#ifdef ESP_BUILD_WITH_CUDA
char* GaussianSplattingRenderer::getOrAllocateScratchBuffer(
    char*& buffer,
    size_t& capacity,
    size_t requiredSize,
    const char* debugName) {
  if (requiredSize == 0) {
    return buffer;
  }

  if (buffer != nullptr && capacity >= requiredSize) {
    return buffer;
  }

  if (buffer != nullptr) {
    cudaFree(buffer);
    buffer = nullptr;
    capacity = 0;
  }

  cudaError_t err = cudaMalloc(&buffer, requiredSize);
  if (err != cudaSuccess) {
    ESP_ERROR() << "Failed to allocate" << debugName << "buffer of size"
                << requiredSize << ":" << cudaGetErrorString(err);
    checkCudaErrors(err);
  }

  capacity = requiredSize;
  ESP_DEBUG() << "Allocated" << debugName << "buffer:" << requiredSize
              << "bytes";
  return buffer;
}

void GaussianSplattingRenderer::ensureDeviceBuffer(
    float*& buffer,
    size_t elementCount,
    const char* debugName) {
  if (buffer) {
    return;
  }

  size_t bytes = elementCount * sizeof(float);
  cudaError_t err = cudaMalloc(&buffer, bytes);
  if (err != cudaSuccess) {
    ESP_ERROR() << "Failed to allocate" << debugName << "buffer of size"
                << bytes << ":" << cudaGetErrorString(err);
    checkCudaErrors(err);
  }
  ESP_DEBUG() << "Allocated" << debugName << "buffer:" << bytes << "bytes";
}
#endif

void GaussianSplattingRenderer::updateAttributes(const float* offsets,
                                                 const float* colors,
                                                 const float* scales) {
#ifndef ESP_BUILD_WITH_CUDA
  ESP_CHECK(false, "GaussianSplattingRenderer::updateAttributes(): CUDA support required");
  return;
#else
  ESP_CHECK(avatarMode_,
            "GaussianSplattingRenderer::updateAttributes(): not in avatar mode");
  if (offsets) {
    checkCudaErrors(cudaMemcpy(d_avatarOffsets_, offsets,
                               numGaussians_ * 3 * sizeof(float),
                               cudaMemcpyDeviceToDevice));
  }
  if (colors) {
    if (shDim_ == 3) {
      checkCudaErrors(cudaMemcpy(d_shs_, colors,
                                 numGaussians_ * 3 * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }
  }
  if (scales) {
    checkCudaErrors(cudaMemcpy(d_scales_, scales,
                               numGaussians_ * 3 * sizeof(float),
                               cudaMemcpyDeviceToDevice));
  }
  if (avatarMode_ && gAvatarBatchCache.valid) {
    gAvatarBatchCache.valid = false;
    gAvatarBatchCache.rendererSignature.clear();
    gAvatarBatchCache.updateSignature.clear();
  }
#endif
}

void GaussianSplattingRenderer::updateAvatar(const float* jointMatrices,
                                             const float* offsets,
                                             const float* colors,
                                             const float* scales) {
#ifndef ESP_BUILD_WITH_CUDA
  ESP_CHECK(false, "GaussianSplattingRenderer::updateAvatar(): CUDA support required");
  return;
#else
  ESP_CHECK(avatarMode_,
            "GaussianSplattingRenderer::updateAvatar(): not in avatar mode");
  ESP_CHECK(jointMatrices,
            "GaussianSplattingRenderer::updateAvatar(): jointMatrices is null");

  updateAttributes(offsets, colors, scales);
  checkCudaErrors(cudaMemcpy(d_avatarJointMatrices_, jointMatrices,
                             avatarJointCount_ * 16 * sizeof(float),
                             cudaMemcpyDeviceToDevice));
  launchAvatarLBS(d_avatarCanonical_, d_avatarOffsets_, d_avatarWeights_,
                  d_avatarJointMatrices_, d_avatarInvBind_,
                  d_avatarCanonicalRotations_, avatarJointCount_, numGaussians_,
                  d_means3D_, d_rotations_, 0);
  checkCudaErrors(cudaGetLastError());
  ++avatarUpdateCalls_;
#endif
}

void GaussianSplattingRenderer::render(Mn::GL::Texture2D* colorTexture,
                                       Mn::GL::Texture2D* depthTexture,
                                       const Mn::Matrix4& viewMatrix,
                                       const Mn::Matrix4& projMatrix,
                                       const Mn::Vector2i& resolution,
                                       const Mn::Vector3& cameraPos,
                                       float nearPlane,
                                       float farPlane,
                                       bool renderColor,
                                       bool renderDepth) {
  render(colorTexture, depthTexture, viewMatrix, projMatrix, resolution,
         cameraPos, nearPlane, farPlane, 0.0f, renderColor, renderDepth);
}

void GaussianSplattingRenderer::render(Mn::GL::Texture2D* colorTexture,
                                       Mn::GL::Texture2D* depthTexture,
                                       const Mn::Matrix4& viewMatrix,
                                       const Mn::Matrix4& projMatrix,
                                       const Mn::Vector2i& resolution,
                                       const Mn::Vector3& cameraPos,
                                       float nearPlane,
                                       float farPlane,
                                       float time,
                                       bool renderColor,
                                       bool renderDepth) {
#ifndef ESP_BUILD_WITH_CUDA
  ESP_CHECK(false, "GaussianSplattingRenderer::render(): CUDA support required");
  return;
#else
  
  ESP_CHECK(initialized_,
            "GaussianSplattingRenderer::render(): Renderer not initialized");
  
  // At least one output must be requested
  ESP_CHECK(renderColor || renderDepth,
            "GaussianSplattingRenderer::render(): At least one of renderColor or renderDepth must be true");
  if (renderDepth) {
    ESP_CHECK(farPlane > nearPlane && nearPlane > 0.0f,
              "GaussianSplattingRenderer::render(): Invalid depth range");
  }
  
  // Check that textures are provided when requested
  ESP_CHECK(!renderColor || colorTexture != nullptr,
            "GaussianSplattingRenderer::render(): colorTexture is null but renderColor is true");
  ESP_CHECK(!renderDepth || depthTexture != nullptr,
            "GaussianSplattingRenderer::render(): depthTexture is null but renderDepth is true");

  const bool avatarBatchPassActive = avatarMode_ && gAvatarBatchContext.active;
  if (avatarBatchPassActive) {
    const auto it = std::find(gAvatarBatchContext.renderers.begin(),
                              gAvatarBatchContext.renderers.end(), this);
    if (it == gAvatarBatchContext.renderers.end()) {
      return;
    }
    if (gAvatarBatchContext.primary != this) {
      return;
    }
  }

  int renderNumGaussians = numGaussians_;
  int renderShDim = shDim_;
  int renderShDegree = shDegree_;
  const float* renderMeans3D = d_means3D_;
  const float* renderShs = d_shs_;
  const float* renderOpacities = d_opacities_;
  const float* renderScales = d_scales_;
  const float* renderRotations = d_rotations_;

  if (avatarBatchPassActive) {
    ESP_CHECK(!gAvatarBatchContext.renderers.empty(),
              "GaussianSplattingRenderer::render(): empty avatar batch");

    std::vector<size_t> updateSignature;
    updateSignature.reserve(gAvatarBatchContext.renderers.size());

    size_t totalGaussians = 0;
    int sharedShDim = gAvatarBatchContext.renderers.front()->shDim_;
    int sharedShDegree = gAvatarBatchContext.renderers.front()->shDegree_;
    for (auto* renderer : gAvatarBatchContext.renderers) {
      ESP_CHECK(renderer && renderer->initialized_ && renderer->avatarMode_,
                "GaussianSplattingRenderer::render(): invalid avatar batch renderer");
      ESP_CHECK(renderer->shDim_ == sharedShDim &&
                    renderer->shDegree_ == sharedShDegree,
                "GaussianSplattingRenderer::render(): avatar batch SH mismatch");
      totalGaussians += static_cast<size_t>(renderer->numGaussians_);
      updateSignature.emplace_back(renderer->avatarUpdateCalls_);
    }

    ESP_CHECK(totalGaussians > 0,
              "GaussianSplattingRenderer::render(): avatar batch has no points");
    ESP_CHECK(totalGaussians <=
                  static_cast<size_t>(std::numeric_limits<int>::max()),
              "GaussianSplattingRenderer::render(): avatar batch too large");

    const bool canReuseCache =
        gAvatarBatchCache.valid &&
        gAvatarBatchCache.totalGaussians == totalGaussians &&
        gAvatarBatchCache.shDim == sharedShDim &&
        gAvatarBatchCache.shDegree == sharedShDegree &&
        gAvatarBatchCache.rendererSignature == gAvatarBatchContext.renderers &&
        gAvatarBatchCache.updateSignature == updateSignature;

    if (!canReuseCache) {
      ensureAvatarBatchBufferCapacity(gAvatarBatchCache.d_means3D,
                                      gAvatarBatchCache.meansCapacity,
                                      totalGaussians * 3, "means");
      ensureAvatarBatchBufferCapacity(gAvatarBatchCache.d_shs,
                                      gAvatarBatchCache.shsCapacity,
                                      totalGaussians *
                                          static_cast<size_t>(sharedShDim),
                                      "shs");
      ensureAvatarBatchBufferCapacity(gAvatarBatchCache.d_opacities,
                                      gAvatarBatchCache.opacitiesCapacity,
                                      totalGaussians, "opacities");
      ensureAvatarBatchBufferCapacity(gAvatarBatchCache.d_scales,
                                      gAvatarBatchCache.scalesCapacity,
                                      totalGaussians * 3, "scales");
      ensureAvatarBatchBufferCapacity(gAvatarBatchCache.d_rotations,
                                      gAvatarBatchCache.rotationsCapacity,
                                      totalGaussians * 4, "rotations");

      size_t gaussianOffset = 0;
      for (auto* renderer : gAvatarBatchContext.renderers) {
        const size_t gaussianCount =
            static_cast<size_t>(renderer->numGaussians_);
        checkCudaErrors(cudaMemcpy(
            gAvatarBatchCache.d_means3D + gaussianOffset * 3,
            renderer->d_means3D_, gaussianCount * 3 * sizeof(float),
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(
            gAvatarBatchCache.d_shs +
                gaussianOffset * static_cast<size_t>(sharedShDim),
            renderer->d_shs_,
            gaussianCount * static_cast<size_t>(sharedShDim) * sizeof(float),
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(
            gAvatarBatchCache.d_opacities + gaussianOffset,
            renderer->d_opacities_, gaussianCount * sizeof(float),
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(
            gAvatarBatchCache.d_scales + gaussianOffset * 3, renderer->d_scales_,
            gaussianCount * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(
            gAvatarBatchCache.d_rotations + gaussianOffset * 4,
            renderer->d_rotations_, gaussianCount * 4 * sizeof(float),
            cudaMemcpyDeviceToDevice));
        gaussianOffset += gaussianCount;
      }
      checkCudaErrors(cudaGetLastError());

      gAvatarBatchCache.valid = true;
      gAvatarBatchCache.rendererSignature = gAvatarBatchContext.renderers;
      gAvatarBatchCache.updateSignature = std::move(updateSignature);
      gAvatarBatchCache.totalGaussians = totalGaussians;
      gAvatarBatchCache.shDim = sharedShDim;
      gAvatarBatchCache.shDegree = sharedShDegree;
    }

    renderNumGaussians = static_cast<int>(totalGaussians);
    renderShDim = sharedShDim;
    renderShDegree = sharedShDegree;
    renderMeans3D = gAvatarBatchCache.d_means3D;
    renderShs = gAvatarBatchCache.d_shs;
    renderOpacities = gAvatarBatchCache.d_opacities;
    renderScales = gAvatarBatchCache.d_scales;
    renderRotations = gAvatarBatchCache.d_rotations;
  }

  int width = resolution.x();
  int height = resolution.y();
  
  // Allocate output buffers conditionally based on what's being rendered
  size_t requiredColorSize = width * height * 4 * sizeof(float);
  size_t requiredDepthSize = width * height * sizeof(float);
  size_t requiredAlphaSize = width * height * sizeof(float);
  
  // Free and reallocate if buffers don't exist or if size doesn't match
  if (lastWidth_ != width || lastHeight_ != height) {
    // Unregister old textures before reallocating
    if (colorInterop_->isRegistered()) {
      colorInterop_.reset(new CudaGLInterop());
    }
    if (depthInterop_->isRegistered()) {
      depthInterop_.reset(new CudaGLInterop());
    }
    
    if (d_outputColor_) cudaFree(d_outputColor_);
    if (d_outputDepth_) cudaFree(d_outputDepth_);
    if (d_outputAlpha_) cudaFree(d_outputAlpha_);
    d_outputColor_ = nullptr;
    d_outputDepth_ = nullptr;
    d_outputAlpha_ = nullptr;
    lastWidth_ = width;
    lastHeight_ = height;
  }
  
  // Only allocate color buffer if rendering color
  if (renderColor && !d_outputColor_) {
    checkCudaErrors(cudaMalloc(&d_outputColor_, requiredColorSize));
    ESP_DEBUG() << "Allocated color buffer:" << requiredColorSize << "bytes";
  }
  
  // Only allocate depth buffer if rendering depth
  if (renderDepth && !d_outputDepth_) {
    checkCudaErrors(cudaMalloc(&d_outputDepth_, requiredDepthSize));
    checkCudaErrors(cudaMemset(d_outputDepth_, 0, requiredDepthSize));
    ESP_DEBUG() << "Allocated depth buffer:" << requiredDepthSize << "bytes";
  }
  if (!d_outputAlpha_) {
    checkCudaErrors(cudaMalloc(&d_outputAlpha_, requiredAlphaSize));
    ESP_DEBUG() << "Allocated alpha buffer:" << requiredAlphaSize << "bytes";
  }

  // Slice 4D Gaussians into per-frame 3D parameters.
  if (foregroundCount_ > 0) {
    ESP_CHECK(d_baseMeans3D_ && d_baseScales_ && d_baseRotations_ &&
                  d_baseOpacities_ && d_times_ && d_timeScales_ && d_motion_,
              "GaussianSplattingRenderer::render(): 4D buffers not initialized");
    launchSliceGaussians(d_baseMeans3D_,
                         d_baseScales_,
                         d_baseRotations_,
                         d_baseOpacities_,
                         d_times_,
                         d_timeScales_,
                         d_motion_,
                         motionStride_,
                         foregroundCount_,
                         time,
                         d_means3D_,
                         d_scales_,
                         d_rotations_,
                         d_opacities_,
                         0);
    checkCudaErrors(cudaGetLastError());
  }
  
  // Convert matrices to arrays - host side
  float h_viewMatrixArray[16];
  float h_projMatrixArray[16];
  
  // Habitat camera transform is camera-to-world. Invert to get world-to-camera.
  Mn::Matrix4 worldToCameraHabitat = viewMatrix.inverted();

  // 3DGS CUDA rasterizer assumes a Y-down, +Z-forward camera coordinate frame.
  // Habitat uses Y-up, -Z-forward. We bridge the two by rotating only the
  // camera basis (world data stays in Habitat coordinates so NavMesh and
  // physics remain consistent).
  const Mn::Matrix4 habitatToGaussian =
      Mn::Matrix4::scaling(Mn::Vector3{1.0f, -1.0f, -1.0f});

  Mn::Matrix4 worldToCameraGaussian =
      habitatToGaussian * worldToCameraHabitat;
  Mn::Matrix4 fullProjHabitat = projMatrix * worldToCameraHabitat;
  // Flip clip-space Y so the CUDA rasterizer works in a Y-down projection.
  // OpenGL/Habitat use Y-up clip coordinates, so we inject the flip here and
  // undo it when writing back to textures inside the CUDA kernels.
  const Mn::Matrix4 clipSpaceYFlip =
      Mn::Matrix4::scaling(Mn::Vector3{1.0f, -1.0f, 1.0f});
  Mn::Matrix4 fullProjGaussian = clipSpaceYFlip * fullProjHabitat;

  matrixToArray(worldToCameraGaussian, h_viewMatrixArray);
  matrixToArray(fullProjGaussian, h_projMatrixArray);
  
  float h_camPos[3] = {cameraPos.x(),
                       cameraPos.y(),
                       cameraPos.z()};
  
  // Debug output for first frame
  static bool debugPrinted = false;
  if (!debugPrinted) {
    ESP_DEBUG() << "=== Camera Debug Info ===";
    ESP_DEBUG() << "Resolution:" << width << "x" << height;
    ESP_DEBUG() << "Num Gaussians:" << renderNumGaussians;
    ESP_DEBUG() << "Camera position:" << cameraPos.x() << "," << cameraPos.y() << "," << cameraPos.z();
    ESP_DEBUG() << "View matrix (camera-to-world):";
    for (int i = 0; i < 4; ++i) {
      ESP_DEBUG() << "  " << viewMatrix[0][i] << " " << viewMatrix[1][i] << " " 
                  << viewMatrix[2][i] << " " << viewMatrix[3][i];
    }
    ESP_DEBUG() << "World-to-camera matrix (Habitat):";
    for (int i = 0; i < 4; ++i) {
      ESP_DEBUG() << "  " << worldToCameraHabitat[0][i] << " " << worldToCameraHabitat[1][i]
                  << " " << worldToCameraHabitat[2][i] << " " << worldToCameraHabitat[3][i];
    }
    ESP_DEBUG() << "World-to-camera matrix (3DGS):";
    for (int i = 0; i < 4; ++i) {
      ESP_DEBUG() << "  " << worldToCameraGaussian[0][i] << " "
                  << worldToCameraGaussian[1][i] << " "
                  << worldToCameraGaussian[2][i] << " "
                  << worldToCameraGaussian[3][i];
    }
    ESP_DEBUG() << "Projection matrix:";
    for (int i = 0; i < 4; ++i) {
      ESP_DEBUG() << "  " << projMatrix[0][i] << " " << projMatrix[1][i] << " " 
                  << projMatrix[2][i] << " " << projMatrix[3][i];
    }
    ESP_DEBUG() << "Full projection matrix (Habitat proj * view):";
    for (int i = 0; i < 4; ++i) {
      ESP_DEBUG() << "  " << fullProjHabitat[0][i] << " " << fullProjHabitat[1][i] << " "
                  << fullProjHabitat[2][i] << " " << fullProjHabitat[3][i];
    }
    debugPrinted = true;
  }
  
  // Allocate persistent buffers for matrices and camera constants once
  ensureDeviceBuffer(d_viewMatrix_, 16, "view matrix");
  ensureDeviceBuffer(d_projMatrix_, 16, "projection matrix");
  ensureDeviceBuffer(d_camPos_, 3, "camera position");
  ensureDeviceBuffer(d_background_, 3, "background color");

  checkCudaErrors(cudaMemcpy(d_viewMatrix_, h_viewMatrixArray,
                             16 * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_projMatrix_, h_projMatrixArray,
                             16 * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_camPos_, h_camPos,
                             3 * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_background_, background_,
                             3 * sizeof(float), cudaMemcpyHostToDevice));
  
  // Calculate FOV from projection matrix
  float tan_fovy = 1.0f / projMatrix[1][1];
  float tan_fovx = 1.0f / projMatrix[0][0];
  
  // Clear any previous CUDA errors
  cudaGetLastError();  // This clears the error flag
  
  if (avatarMode_) {
    ++avatarRenderCalls_;
    if (avatarUpdateCalls_ == avatarLastRenderedUpdateCall_) {
      ++avatarRendersReusingPose_;
    }
    avatarLastRenderedUpdateCall_ = avatarUpdateCalls_;
  }

  // Call CUDA rasterizer.
  auto geometryAllocator = [this](size_t size) -> char* {
    return getOrAllocateScratchBuffer(
        geometryScratchBuffer_, geometryScratchCapacity_, size, "geometry");
  };
  auto binningAllocator = [this](size_t size) -> char* {
    return getOrAllocateScratchBuffer(
        binningScratchBuffer_, binningScratchCapacity_, size, "binning");
  };
  auto imageAllocator = [this](size_t size) -> char* {
    return getOrAllocateScratchBuffer(
        imageScratchBuffer_, imageScratchCapacity_, size, "image");
  };
  
  const int shDegree = renderShDegree;
  const int shCoeffCount = renderShDim / 3;
  const float* colorsPrecomp = nullptr;
  const float* shsPtr = renderShs;

  const bool antialiasing = !avatarMode_;

  CudaRasterizer::Rasterizer::forward(
      geometryAllocator,
      binningAllocator,
      imageAllocator,
      renderNumGaussians,  // P: number of Gaussians
      shDegree,       // D: SH degree
      shCoeffCount,   // M: number of SH coefficients (in vec3 units)
      d_background_,  // background color (device memory)
      width,
      height,
      renderMeans3D,
      shsPtr,
      colorsPrecomp,
      renderOpacities,
      renderScales,
      1.0f,           // scale_modifier
      renderRotations,
      nullptr,        // cov3D_precomp (not used)
      d_viewMatrix_,   // device memory
      d_projMatrix_,   // device memory
      d_camPos_,       // device memory
      tan_fovx,
      tan_fovy,
      false,          // prefiltered
      renderColor ? d_outputColor_ : nullptr,  // output color (nullptr if not rendering color)
      renderDepth ? d_outputDepth_ : nullptr,  // output depth (nullptr if not rendering depth)
      d_outputAlpha_,                          // output alpha
      antialiasing,                           // antialiasing
      nullptr,
      false
  );

  // Check for CUDA errors after rasterizer call
  cudaError_t rastErr = cudaGetLastError();
  if (rastErr != cudaSuccess) {
    ESP_ERROR() << "CUDA error after rasterizer forward!";
    ESP_ERROR() << "Error:" << cudaGetErrorString(rastErr);
    checkCudaErrors(rastErr);
  }
  
  // Register textures if not already registered (only for requested outputs)
  if (renderColor) {
    colorInterop_->registerTexture(*colorTexture, colorTexture->id());
  }
  
  if (renderDepth) {
    depthInterop_->registerTexture(*depthTexture, depthTexture->id());
  }
  
  // Map OpenGL textures to CUDA (only for requested outputs)
  cudaSurfaceObject_t colorSurface = 0;
  cudaSurfaceObject_t depthSurface = 0;
  if (renderColor) {
    colorInterop_->mapResources();
  }
  if (renderDepth) {
    depthInterop_->mapResources();
  }
  
  // Process and write color data directly to OpenGL texture (only if requested)
  if (renderColor) {
    cudaArray_t colorArray = colorInterop_->getMappedArray();

    colorSurface = createSurfaceObject(colorArray);
    if (colorSurface == 0) {
      ESP_ERROR() << "Failed to create CUDA surface for color texture";
    } else {
      // Launch CUDA kernel to convert planar RGB to interleaved RGBA,
      // flip horizontally, and write directly to OpenGL texture
      // This avoids allocating temporary buffers and extra memory copies
      launchPlanarRGBToInterleavedRGBASurface(
          d_outputColor_,  // Input: planar RGB from rasterizer
          d_outputAlpha_,  // Input: alpha from rasterizer
          colorSurface,    // Output surface for OpenGL texture
          width,
          height,
          0);              // Default stream
      
      // Check for kernel launch errors
      cudaError_t kernelErr = cudaGetLastError();
      if (kernelErr != cudaSuccess) {
        ESP_ERROR() << "CUDA kernel launch error (RGB conversion):"
                    << cudaGetErrorString(kernelErr);
        checkCudaErrors(kernelErr);
      }
    }
  }
  
  // Process and write depth data directly to OpenGL texture (only if requested)
  if (renderDepth) {
    cudaArray_t depthArray = depthInterop_->getMappedArray();

    depthSurface = createSurfaceObject(depthArray);
    if (depthSurface == 0) {
      ESP_ERROR() << "Failed to create CUDA surface for depth texture";
    } else {
      // Avoid writing depth for low-alpha avatar splats to prevent halos.
      constexpr float kAvatarDepthAlphaCutoff = 0.6f;
      constexpr float kSceneDepthAlphaCutoff = 0.01f;
      const float depthAlphaCutoff =
          avatarMode_ ? kAvatarDepthAlphaCutoff : kSceneDepthAlphaCutoff;
      // Launch CUDA kernel to flip depth horizontally and write directly to OpenGL texture
      launchDepthToSurface(
          d_outputDepth_,  // Input: depth from rasterizer
          d_outputAlpha_,  // Input: alpha from rasterizer
          depthSurface,    // Output surface for OpenGL texture
          width,
          height,
          nearPlane,
          farPlane,
          depthAlphaCutoff,
          0);              // Default stream
      
      // Check for kernel launch errors
      cudaError_t depthKernelErr = cudaGetLastError();
      if (depthKernelErr != cudaSuccess) {
        ESP_ERROR() << "CUDA kernel launch error (depth flip):"
                    << cudaGetErrorString(depthKernelErr);
        checkCudaErrors(depthKernelErr);
      }
    }
  }
  
  // Unmap resources to make them available to OpenGL (only for requested outputs)
  if (renderColor) {
    colorInterop_->unmapResources();
  }
  if (renderDepth) {
    depthInterop_->unmapResources();
  }

  // Destroy surface objects after CUDA-GL synchronization has completed via
  // cudaGraphicsUnmapResources.
  if (colorSurface != 0) {
    destroySurfaceObject(colorSurface);
  }
  if (depthSurface != 0) {
    destroySurfaceObject(depthSurface);
  }
#endif
}

}  // namespace gfx
}  // namespace esp
