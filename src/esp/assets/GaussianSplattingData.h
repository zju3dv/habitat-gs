#ifndef ESP_ASSETS_GAUSSIANSPLATTINGDATA_H_
#define ESP_ASSETS_GAUSSIANSPLATTINGDATA_H_

/** @file
 * @brief Class @ref esp::assets::GaussianSplattingData
 */

#include <Corrade/Containers/Array.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector3.h>

#include "BaseMesh.h"
#include "esp/core/Esp.h"

namespace Mn = Magnum;

namespace esp {
namespace assets {

/**
 * @brief Base structure representing a single 3D Gaussian Splat.
 */
struct GaussianSplat {
  //! Position of the Gaussian center
  Mn::Vector3 position;

  //! Normal vector (nx, ny, nz)
  Mn::Vector3 normal;

  //! Spherical harmonics DC coefficients (RGB base color)
  Mn::Vector3 f_dc;  // f_dc_0, f_dc_1, f_dc_2

  //! Spherical harmonics higher-order coefficients (45 floats for degree 3)
  //! f_rest_0 to f_rest_44
  Corrade::Containers::Array<float> f_rest;

  //! Opacity of the Gaussian
  float opacity;

  //! Scale of the Gaussian in 3 axes
  Mn::Vector3 scale;  // scale_0, scale_1, scale_2

  //! Rotation quaternion (rot_0, rot_1, rot_2, rot_3)
  Mn::Quaternion rotation;

  //! Optional right-hand rotation quaternion (used by some 4D assets)
  Mn::Quaternion rotationR{Mn::Vector3{0.0f}, 1.0f};

  //! Whether this splat carries a secondary rotation
  bool hasRotationR = false;

  /**
   * @brief Default constructor
   */
  GaussianSplat() : opacity(1.0f) {}

  /**
   * @brief Move constructor (required because Array is move-only)
   */
  GaussianSplat(GaussianSplat&&) noexcept = default;

  /**
   * @brief Move assignment operator (required because Array is move-only)
   */
  GaussianSplat& operator=(GaussianSplat&&) noexcept = default;

  /**
   * @brief Deleted copy constructor (cannot copy Array)
   */
  GaussianSplat(const GaussianSplat&) = delete;

  /**
   * @brief Deleted copy assignment (cannot copy Array)
   */
  GaussianSplat& operator=(const GaussianSplat&) = delete;
};

/**
 * @brief 4D Gaussian splat with temporal and motion attributes.
 *
 * Extends the base 3D splat with time, temporal scale, and motion terms.
 */
struct GaussianSplat4D : public GaussianSplat {
  //! Temporal center (t) of the 4D Gaussian
  float time = 0.0f;

  //! Temporal scale (log-space, analogous to spatial scale)
  float timeScale = 0.0f;

  //! Linear motion vector (vx, vy, vz) for 4D slicing
  Mn::Vector3 motion{0.0f, 0.0f, 0.0f};

  //! Optional acceleration term for higher-order motion
  Mn::Vector3 motionAccel{0.0f, 0.0f, 0.0f};

  //! Optional jerk term for higher-order motion
  Mn::Vector3 motionJerk{0.0f, 0.0f, 0.0f};

  //! Number of motion coefficients stored (3 for velocity-only, 9 for up to jerk)
  int motionDim = 3;
};

/**
 * @brief Data storage class for 3D Gaussian Splatting assets.
 *
 * This class stores and manages 3D Gaussian Splatting data loaded from PLY
 * files. Unlike traditional mesh data, Gaussian Splatting represents scenes
 * as a collection of oriented 3D Gaussians.
 */
class GaussianSplattingData : public BaseMesh {
 public:
  /**
   * @brief Supported Gaussian asset dimensionality.
   */
  enum class Layout { k3D, k4D, kHybrid };

  /**
   * @brief Rendering buffer structure for GPU upload.
   */
  struct RenderingBuffer {
    //! Buffer containing Gaussian positions
    Mn::GL::Buffer positionBuffer;
    //! Buffer containing Gaussian normals
    Mn::GL::Buffer normalBuffer;
    //! Buffer containing spherical harmonics DC coefficients
    Mn::GL::Buffer shDCBuffer;
    //! Buffer containing spherical harmonics rest coefficients
    Mn::GL::Buffer shRestBuffer;
    //! Buffer containing opacity values
    Mn::GL::Buffer opacityBuffer;
    //! Buffer containing scale values
    Mn::GL::Buffer scaleBuffer;
    //! Buffer containing rotation quaternions
    Mn::GL::Buffer rotationBuffer;
  };

  /**
   * @brief Constructor. Sets asset type to GAUSSIAN_SPLATTING.
   */
  explicit GaussianSplattingData()
      : BaseMesh(SupportedMeshType::GAUSSIAN_SPLATTING) {}

  /**
   * @brief Destructor
   */
  ~GaussianSplattingData() override = default;

  /**
   * @brief Upload Gaussian data to GPU buffers.
   * @param forceReload If true, recompiles the buffers even if already uploaded.
   */
  void uploadBuffersToGPU(bool forceReload = false) override;

  /**
   * @brief Add a static (3D) Gaussian splat to the data.
   * @param splat The Gaussian splat to add (will be moved).
   */
  void addGaussian(GaussianSplat&& splat);

  /**
   * @brief Add a static (3D) Gaussian splat to the data.
   */
  void addStaticGaussian(GaussianSplat&& splat);

  /**
   * @brief Add a dynamic (4D) Gaussian splat to the data.
   */
  void addDynamicGaussian(GaussianSplat4D&& splat);

  /**
   * @brief Get the number of Gaussian splats.
   * @return Number of Gaussians stored.
   */
  size_t getGaussianCount() const {
    return gaussians3D_.size() + gaussians4D_.size();
  }

  /**
   * @brief Get the number of static/background Gaussians.
   */
  size_t getStaticGaussianCount() const { return gaussians3D_.size(); }

  /**
   * @brief Get the number of dynamic/foreground Gaussians.
   */
  size_t getDynamicGaussianCount() const { return gaussians4D_.size(); }

  /**
   * @brief Get read-only access to Gaussian data.
   * @return Const reference to the vector of Gaussians.
   */
  const std::vector<GaussianSplat>& getGaussians() const;

  /**
   * @brief Get read-only access to static/background Gaussians.
   */
  const std::vector<GaussianSplat>& getStaticGaussians() const {
    return gaussians3D_;
  }

  /**
   * @brief Get read-only access to dynamic/foreground Gaussians.
   */
  const std::vector<GaussianSplat4D>& getDynamicGaussians() const {
    return gaussians4D_;
  }

  /**
   * @brief Get the rendering buffer (for GPU rendering).
   * @return Pointer to the rendering buffer, or nullptr if not uploaded.
   */
  RenderingBuffer* getRenderingBuffer() {
    return renderingBuffer_.get();
  }

  /**
   * @brief Reserve space for a given number of Gaussians.
   * @param count Number of Gaussians to reserve space for.
   */
  void reserve(size_t count) { gaussians3D_.reserve(count); }

  /**
   * @brief Clear all Gaussian data.
   */
  void clear() {
    gaussians3D_.clear();
    gaussians4D_.clear();
    combinedGaussians_.clear();
    combinedDirty_ = true;
    buffersOnGPU_ = false;
    renderingBuffer_.reset();
    layout_ = Layout::k3D;
    motionStride_ = 3;
    hasRotationR_ = false;
    shRestCountStatic_ = 0;
    shRestCountDynamic_ = 0;
    maxShRestCount_ = 0;
  }

  /**
   * @brief Get the number of spherical harmonics coefficients per Gaussian.
   * @return Number of SH rest coefficients (typically 45 for degree 3).
   */
  size_t getSHRestCount() const { return maxShRestCount_; }

  /**
   * @brief Get SH rest coefficient count for dynamic/foreground Gaussians.
   */
  size_t getDynamicSHRestCount() const { return shRestCountDynamic_; }

  /**
   * @brief Get SH rest coefficient count for static/background Gaussians.
   */
  size_t getStaticSHRestCount() const { return shRestCountStatic_; }

  /**
   * @brief Get the number of motion coefficients stored per Gaussian.
   */
  int getMotionStride() const { return motionStride_; }

  /**
   * @brief Whether a secondary rotation quaternion is present.
   */
  bool hasRotationR() const { return hasRotationR_; }

  /**
   * @brief Mark whether this dataset represents 4D Gaussians.
   */
  void setLayout(Layout layout) { layout_ = layout; }

  /**
   * @brief Check if this dataset includes 4D Gaussians.
   */
  bool is4D() const { return layout_ == Layout::k4D || layout_ == Layout::kHybrid; }

  /**
   * @brief Check if this dataset mixes foreground (4D) and background (3D).
   */
  bool isHybrid() const { return layout_ == Layout::kHybrid; }

 protected:
  //! Vector storing static/background Gaussian splats
  std::vector<GaussianSplat> gaussians3D_;

  //! Vector storing dynamic/foreground Gaussian splats
  std::vector<GaussianSplat4D> gaussians4D_;

  //! Combined cache returned by getGaussians() for legacy callers
  mutable std::vector<GaussianSplat> combinedGaussians_;
  mutable bool combinedDirty_ = true;

  //! Rendering buffer for GPU upload
  std::unique_ptr<RenderingBuffer> renderingBuffer_ = nullptr;

  //! Dimensionality tag for renderer/importer integration
  Layout layout_ = Layout::k3D;

  //! Motion coefficient stride (3 for velocity-only, 9 if accel/jerk present)
  int motionStride_ = 3;

  //! Whether secondary rotations are populated
  bool hasRotationR_ = false;

  //! Max SH rest count across all gaussians
  size_t maxShRestCount_ = 0;

  //! SH rest count for static/background gaussians
  size_t shRestCountStatic_ = 0;

  //! SH rest count for dynamic/foreground gaussians
  size_t shRestCountDynamic_ = 0;
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_GAUSSIANSPLATTINGDATA_H_
