#ifndef ESP_ASSETS_GAUSSIANSPLATTINGIMPORTER_H_
#define ESP_ASSETS_GAUSSIANSPLATTINGIMPORTER_H_

/** @file
 * @brief Class @ref esp::assets::GaussianSplattingImporter
 */

#include <Corrade/Containers/Optional.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Trade/AbstractImporter.h>

#include <array>
#include <fstream>
#include <string>
#include <vector>

// Needed in the header because private helper declarations use mz_zip_archive,
// and the current miniz typedef is an anonymous struct that can't be forward-declared.
// Only pull in declarations here; the implementation stays in the .cpp file.
#define MINIZ_HEADER_FILE_ONLY
#include "deps/assimp/contrib/zip/src/miniz.h"
#undef MINIZ_HEADER_FILE_ONLY

namespace Mn = Magnum;

namespace esp {
namespace assets {

/**
 * @brief Custom importer for 3D Gaussian Splatting PLY files.
 *
 * This importer extends Magnum::Trade::AbstractImporter to support loading
 * 3DGS PLY files with the specific format containing Gaussian parameters
 * (position, normal, spherical harmonics, opacity, scale, rotation).
 */
class GaussianSplattingImporter : public Mn::Trade::AbstractImporter {
 public:
  struct GaussianSetData {
    size_t count = 0;
    size_t shRestCount = 0;
    int activeShDegree = 3;
    int activeShDegreeT = 0;
    bool hasRotationR = false;
    bool hasMotion9 = false;
    int motionStride = 3;
    std::vector<Magnum::Vector3> positions;
    std::vector<Magnum::Vector3> normals;
    std::vector<Magnum::Vector3> sh_dc;
    std::vector<std::vector<float>> sh_rest;
    std::vector<float> opacities;
    std::vector<Magnum::Vector3> scales;
    std::vector<Magnum::Quaternion> rotations;
    std::vector<Magnum::Quaternion> rotationsR;
    std::vector<float> times;
    std::vector<float> timeScales;
    std::vector<Magnum::Vector3> motions;
    std::vector<std::array<float, 9>> motionCoeffs;
    std::vector<std::vector<float>> semantics;
    size_t semanticDim = 0;
  };

  struct PropertyInfo {
    enum class Type { Float, UChar };
    std::string name;
    Type type = Type::Float;
    size_t offset = 0;
    size_t size = 0;
  };

  /**
   * @brief Constructor
   */
  explicit GaussianSplattingImporter();

  /**
   * @brief Destructor
   */
  ~GaussianSplattingImporter() override = default;

  /**
   * @brief Get importer features
   */
  Mn::Trade::ImporterFeatures doFeatures() const override;

  /**
   * @brief Check if the file is a 3DGS PLY file
   */
  bool doIsOpened() const override;

  /**
   * @brief Open a 3DGS PLY file from the filesystem
   * @param filename Path to the PLY file
   */
  void doOpenFile(Corrade::Containers::StringView filename) override;

  /**
   * @brief Close the currently opened file
   */
  void doClose() override;

  /**
   * @brief Get the number of meshes (always 1 for 3DGS data)
   */
  Mn::UnsignedInt doMeshCount() const override;

  /**
   * @brief Load mesh data at given index (converts Gaussians to point cloud)
   * @param id Mesh index (should be 0)
   */
  Corrade::Containers::Optional<Mn::Trade::MeshData> doMesh(
      Mn::UnsignedInt id,
      Mn::UnsignedInt level) override;

  /**
   * @brief Get the loaded Gaussian count
   */
  size_t getGaussianCount() const { return gaussianCount_; }

  /**
   * @brief Get foreground (4D) Gaussian count for hybrid assets.
   */
  size_t getForegroundCount() const { return foregroundData_.count; }

  /**
   * @brief Get background (3D) Gaussian count for hybrid assets.
   */
  size_t getBackgroundCount() const { return backgroundData_.count; }

  /**
   * @brief Get all loaded Gaussian positions
   */
  const std::vector<Magnum::Vector3>& getPositions() const { 
    return positions_; 
  }

  /**
   * @brief Get all loaded Gaussian normals
   */
  const std::vector<Magnum::Vector3>& getNormals() const { 
    return normals_; 
  }

  /**
   * @brief Get all loaded Gaussian SH DC coefficients
   */
  const std::vector<Magnum::Vector3>& getSHDC() const { 
    return sh_dc_; 
  }

  /**
   * @brief Get all loaded Gaussian SH rest coefficients
   */
  const std::vector<std::vector<float>>& getSHRest() const { 
    return sh_rest_; 
  }

  /**
   * @brief Get all loaded Gaussian opacities
   */
  const std::vector<float>& getOpacities() const { 
    return opacities_; 
  }

  /**
   * @brief Get all loaded Gaussian scales
   */
  const std::vector<Magnum::Vector3>& getScales() const { 
    return scales_; 
  }

  /**
   * @brief Get all loaded Gaussian rotations
   */
  const std::vector<Magnum::Quaternion>& getRotations() const { 
    return rotations_; 
  }

  /**
   * @brief Get secondary rotations (only populated for .4dgs.npz)
   */
  const std::vector<Magnum::Quaternion>& getRotationsR() const {
    return rotationsR_;
  }

  /**
   * @brief Get temporal centers per Gaussian (only set for 4D assets).
   */
  const std::vector<float>& getTimes() const { return times_; }

  /**
   * @brief Get temporal scales per Gaussian (only set for 4D assets).
   */
  const std::vector<float>& getTimeScales() const { return timeScales_; }

  /**
   * @brief Get motion vectors per Gaussian (only set for 4D assets).
   */
  const std::vector<Magnum::Vector3>& getMotions() const { return motions_; }

  /**
   * @brief Get higher-order motion coefficients per Gaussian (stride 9).
   */
  const std::vector<std::array<float, 9>>& getMotionCoeffs() const {
    return motionCoeffs_;
  }

  /**
   * @brief Per-Gaussian semantic embedding (foreground/background).
   */
  const std::vector<std::vector<float>>& getForegroundSemantics() const {
    return foregroundData_.semantics;
  }
  const std::vector<std::vector<float>>& getBackgroundSemantics() const {
    return backgroundData_.semantics;
  }

  /**
   * @brief HDBSCAN cluster cache used to group semantics.
   */
  const std::vector<std::vector<float>>& getHdbscanCache() const {
    return hdbscanCache_;
  }

  /**
   * @brief Temporal metadata (seconds) from NPZ if available.
   */
  float getDuration() const { return duration_; }
  float getInterval() const { return interval_; }
  float getForegroundInterval() const { return foregroundInterval_; }

  /**
   * @brief Whether the currently opened asset contains 4D attributes.
   */
  bool is4D() const { return is4D_; }

  /**
   * @brief Whether the asset mixes foreground (4D) and background (3D).
   */
  bool isHybrid() const { return isHybrid_; }

  /**
   * @brief Whether motion contains 9 coefficients (velocity/accel/jerk).
   */
  bool hasMotion9() const { return hasMotion9_; }

  /**
   * @brief Foreground (dynamic) data block.
   */
  const GaussianSetData& getForegroundData() const { return foregroundData_; }

  /**
   * @brief Background (static) data block.
   */
  const GaussianSetData& getBackgroundData() const { return backgroundData_; }

 private:
  /**
   * @brief Parse PLY header and determine property layout
   */
  bool parsePLYHeader(std::ifstream& file);

  /**
   * @brief Read binary PLY data
   */
  bool readBinaryData(std::ifstream& file);

  /**
   * @brief Load NPZ-formatted 4DGS data.
   */
  bool loadFromNpz(const std::string& filename);

  /**
   * @brief Load InteractiveStreet foreground/background NPZ data.
   */
  bool loadInteractiveNpz(mz_zip_archive& zip, const std::string& filename);

  /**
   * @brief Load standard (single-stream) 4DGS NPZ data.
   */
  bool loadTemporalNpz(mz_zip_archive& zip, const std::string& filename);

  /**
   * @brief Convert little-endian bytes to float
   */
  float bytesToFloat(const unsigned char* bytes) const;

  /**
   * @brief Rebuild combined buffers from foreground/background sets.
   */
  void rebuildCombinedBuffers();

  //! Property layout for vertices
  std::vector<PropertyInfo> properties_;

  //! Size of one vertex entry in bytes
  size_t bytesPerVertex_ = 0;

  //! Whether a file is currently opened
  bool opened_ = false;

  //! Number of Gaussians loaded
  size_t gaussianCount_ = 0;

  //! Number of SH rest coefficients per Gaussian (max across sets)
  size_t shRestCount_ = 0;

  //! Loaded Gaussian positions
  std::vector<Magnum::Vector3> positions_;

  //! Loaded Gaussian normals
  std::vector<Magnum::Vector3> normals_;

  //! Loaded Gaussian SH DC coefficients (base color)
  std::vector<Magnum::Vector3> sh_dc_;

  //! Loaded Gaussian SH rest coefficients
  std::vector<std::vector<float>> sh_rest_;

  //! Loaded Gaussian opacities
  std::vector<float> opacities_;

  //! Loaded Gaussian scales
  std::vector<Magnum::Vector3> scales_;

  //! Loaded Gaussian rotations (quaternions)
  std::vector<Magnum::Quaternion> rotations_;

  //! Loaded secondary rotations (4D right-hand)
  std::vector<Magnum::Quaternion> rotationsR_;

  //! Loaded Gaussian temporal centers
  std::vector<float> times_;

  //! Loaded Gaussian temporal scales (log-space)
  std::vector<float> timeScales_;

  //! Loaded Gaussian motion vectors
  std::vector<Magnum::Vector3> motions_;

  //! Optional higher-order motion coefficients (stride 9)
  std::vector<std::array<float, 9>> motionCoeffs_;

  //! Whether temporal attributes were present
  bool is4D_ = false;

  //! Whether motion contains 9 coefficients
  bool hasMotion9_ = false;

  //! Whether the dataset mixes foreground/background
  bool isHybrid_ = false;

  //! Foreground (dynamic) set data
  GaussianSetData foregroundData_;

  //! Background (static) set data
  GaussianSetData backgroundData_;

  //! Cached HDBSCAN cluster centers (K x D)
  std::vector<std::vector<float>> hdbscanCache_;

  //! Temporal metadata
  float duration_ = 0.0f;
  float interval_ = 0.0f;
  float foregroundInterval_ = 0.0f;
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_GAUSSIANSPLATTINGIMPORTER_H_
