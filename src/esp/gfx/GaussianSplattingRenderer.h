#ifndef ESP_GFX_GAUSSIANSPLATTINGRENDERER_H_
#define ESP_GFX_GAUSSIANSPLATTINGRENDERER_H_

#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector.h>
#include <memory>
#include <vector>

#include "esp/assets/GaussianSplattingData.h"
#include "esp/assets/GaussianAvatarData.h"
#include "esp/core/Esp.h"
#include "esp/gfx/configure.h"

#ifdef ESP_BUILD_WITH_CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#endif

namespace esp {
namespace gfx {

/**
 * @brief CUDA-GL interop helper for managing OpenGL texture registration
 */
class CudaGLInterop {
 public:
  CudaGLInterop() = default;
  ~CudaGLInterop();

  /**
   * @brief Register an OpenGL texture for CUDA access
   */
  void registerTexture(Magnum::GL::Texture2D& texture, unsigned int glTextureId);

  /**
   * @brief Map the registered texture to get CUDA surface object
   */
  void mapResources();

  /**
   * @brief Unmap the resources
   */
  void unmapResources();

  /**
   * @brief Get CUDA array for writing
   */
#ifdef ESP_BUILD_WITH_CUDA
  cudaArray_t getMappedArray();
#endif

  /**
   * @brief Check if texture is registered
   */
  bool isRegistered() const { return registered_; }

  /**
   * @brief Best-effort release of the registered CUDA-GL resource.
   *
   * This is safe to call multiple times and is intentionally tolerant of
   * shutdown ordering where the backing GL texture may already be gone.
   */
  void release();

  /**
   * @brief Get currently registered GL texture id (0 if none)
   */
  unsigned int registeredTextureId() const {
#ifdef ESP_BUILD_WITH_CUDA
    return registeredTextureId_;
#else
    return 0;
#endif
  }

 private:
#ifdef ESP_BUILD_WITH_CUDA
  cudaGraphicsResource_t cudaResource_ = nullptr;
  unsigned int registeredTextureId_ = 0;
#endif
  bool registered_ = false;
  bool mapped_ = false;
};

/**
 * @brief Renderer for 3D Gaussian Splatting using CUDA
 *
 * This class manages CUDA-based rasterization of Gaussian splats and
 * provides CUDA-OpenGL interoperability to render directly into OpenGL
 * textures.
 */
class GaussianSplattingRenderer {
 public:
  /**
   * @brief Constructor
   */
  GaussianSplattingRenderer();

  /**
   * @brief Destructor
   */
  ~GaussianSplattingRenderer();

  /**
   * @brief Initialize the renderer with Gaussian data
   */
  void initialize(const assets::GaussianSplattingData& gaussianData, bool normalizeQuaternion = true);

  /**
   * @brief Initialize the renderer with Gaussian Avatar data.
   */
  void initializeAvatar(const assets::GaussianAvatarData& avatarData);

  /**
   * @brief Update avatar attributes from device pointers.
   */
  void updateAttributes(const float* offsets,
                        const float* colors,
                        const float* scales);

  /**
   * @brief Update avatar pose and attributes from device pointers.
   */
  void updateAvatar(const float* jointMatrices,
                    const float* offsets,
                    const float* colors,
                    const float* scales);

  /**
   * @brief Render Gaussians to CUDA-mapped OpenGL textures
   *
   * @param colorTexture OpenGL color texture to render into (can be nullptr if renderColor=false)
   * @param depthTexture OpenGL depth texture to render into (can be nullptr if renderDepth=false)
   * @param viewMatrix Camera view matrix
   * @param projMatrix Camera projection matrix
   * @param resolution Output resolution (width, height)
   * @param cameraPos Camera position in world space
   * @param time Time cursor for 4D Gaussian slicing (seconds)
   * @param renderColor Whether to render color output
   * @param renderDepth Whether to render depth output
   */
  void render(Magnum::GL::Texture2D* colorTexture,
              Magnum::GL::Texture2D* depthTexture,
              const Magnum::Matrix4& viewMatrix,
              const Magnum::Matrix4& projMatrix,
              const Magnum::Vector2i& resolution,
              const Magnum::Vector3& cameraPos,
              float nearPlane,
              float farPlane,
              float time,
              bool renderColor = true,
              bool renderDepth = true);

  /**
   * @brief Compatibility overload without explicit time; dispatches with time
   * set to 0.
   */
  void render(Magnum::GL::Texture2D* colorTexture,
              Magnum::GL::Texture2D* depthTexture,
              const Magnum::Matrix4& viewMatrix,
              const Magnum::Matrix4& projMatrix,
              const Magnum::Vector2i& resolution,
              const Magnum::Vector3& cameraPos,
              float nearPlane,
              float farPlane,
              bool renderColor = true,
              bool renderDepth = true);

  /**
   * @brief Check if renderer is initialized
   */
  bool isInitialized() const { return initialized_; }

  /**
   * @brief Release any CUDA-GL interop registrations held by this renderer.
   *
   * Call this before destroying the render targets that back the registered
   * textures.
   */
  void releaseInteropResources();

 private:
  /**
   * @brief Prepare GPU buffers for rendering
   */
  void prepareGPUBuffers();

  /**
   * @brief Free GPU buffers
   */
  void freeGPUBuffers();

  /**
   * @brief Convert Magnum matrix to flat array for CUDA
   */
  void matrixToArray(const Magnum::Matrix4& matrix, float* array);

  bool initialized_ = false;
  int lastWidth_ = 0;
  int lastHeight_ = 0;

  // GPU data pointers
#ifdef ESP_BUILD_WITH_CUDA
  float* d_means3D_ = nullptr;      // Gaussian positions
  float* d_shs_ = nullptr;           // Spherical harmonics
  float* d_opacities_ = nullptr;     // Opacities
  float* d_scales_ = nullptr;        // Scales
  float* d_rotations_ = nullptr;     // Rotations (quaternions)
  float* d_avatarCanonical_ = nullptr;   // Avatar canonical means
  float* d_avatarWeights_ = nullptr;     // Avatar skinning weights
  float* d_avatarOffsets_ = nullptr;     // Avatar offsets (optional)
  float* d_avatarCanonicalRotations_ = nullptr;  // Avatar canonical rotations
  float* d_avatarJointMatrices_ = nullptr;  // Avatar joint matrices
  float* d_avatarInvBind_ = nullptr;     // Avatar inverse bind matrices
  // 4D base buffers (remain constant across frames)
  float* d_baseMeans3D_ = nullptr;
  float* d_baseOpacities_ = nullptr;
  float* d_baseScales_ = nullptr;
  float* d_baseRotations_ = nullptr;
  float* d_times_ = nullptr;
  float* d_timeScales_ = nullptr;
  float* d_motion_ = nullptr;
  float* d_viewMatrix_ = nullptr;    // View matrix (4x4)
  float* d_projMatrix_ = nullptr;    // Projection matrix (4x4)
  float* d_camPos_ = nullptr;        // Camera position (x, y, z)
  float* d_background_ = nullptr;    // Background color (rgb)
  
  // Temporary output buffers
  float* d_outputColor_ = nullptr;   // RGBA output
  float* d_outputDepth_ = nullptr;   // Depth output
  float* d_outputAlpha_ = nullptr;   // Alpha output
  int* d_debugRadii_ = nullptr;      // Optional debug radii buffer

  // Persistent scratch buffers used by CUDA rasterizer
  char* geometryScratchBuffer_ = nullptr;
  char* binningScratchBuffer_ = nullptr;
  char* imageScratchBuffer_ = nullptr;
  size_t geometryScratchCapacity_ = 0;
  size_t binningScratchCapacity_ = 0;
  size_t imageScratchCapacity_ = 0;

  char* getOrAllocateScratchBuffer(char*& buffer,
                                   size_t& capacity,
                                   size_t requiredSize,
                                   const char* debugName);
  void ensureDeviceBuffer(float*& buffer,
                          size_t elementCount,
                          const char* debugName);
  
  // CUDA-GL interop helpers
  std::unique_ptr<CudaGLInterop> colorInterop_;
  std::unique_ptr<CudaGLInterop> depthInterop_;
#endif

  int numGaussians_ = 0;
  int foregroundCount_ = 0;
  int backgroundCount_ = 0;
  int shDegree_ = 3;  // Spherical harmonics degree
  int shDim_ = 16;    // SH dimensions: (degree + 1)^2
  int motionStride_ = 3;  // Motion coefficients per Gaussian
  int avatarJointCount_ = 0;
  bool avatarMode_ = false;
  size_t avatarUpdateCalls_ = 0;
  size_t avatarRenderCalls_ = 0;
  size_t avatarRendersReusingPose_ = 0;
  size_t avatarLastRenderedUpdateCall_ = 0;

  // Background color
  float background_[3] = {0.0f, 0.0f, 0.0f};
  bool is4D_ = false;
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANSPLATTINGRENDERER_H_
