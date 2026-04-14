#include "GaussianSplattingImporter.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/String.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Trade/MeshData.h>

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

#include "esp/core/Esp.h"
// Use the miniz implementation bundled with assimp for NPZ zip handling.
#include "deps/assimp/contrib/zip/src/miniz.h"

namespace Cr = Corrade;
namespace Mn = Magnum;

namespace esp {
namespace assets {

namespace {
bool isGaussianRestProperty(const std::string& name) {
  return name.rfind("f_rest_", 0) == 0;
}

struct NpyArrayData {
  std::vector<size_t> shape;
  char typeChar = 0;
  size_t wordSize = 0;
  bool fortran = false;
  std::vector<unsigned char> data;

  size_t elementCount() const {
    size_t count = 1;
    for (size_t d : shape) {
      count *= d;
    }
    return count;
  }
};

std::string ltrim(const std::string& s) {
  const auto pos = s.find_first_not_of(" \t\n\r");
  return pos == std::string::npos ? "" : s.substr(pos);
}

bool parseNpyHeader(const std::string& header, NpyArrayData& out) {
  const auto descrPos = header.find("descr");
  if (descrPos == std::string::npos) {
    return false;
  }
  // The header looks like: {'descr': '<f4', 'fortran_order': False, ...}
  // Skip the quoted key and grab the quoted dtype value.
  const auto afterKey = header.find("descr':", descrPos);
  const auto quoteKeyEnd = header.find('\'', afterKey == std::string::npos
                                                ? descrPos
                                                : afterKey);
  const auto quoteValBeg = header.find('\'', quoteKeyEnd + 1);
  const auto quoteValEnd = header.find('\'', quoteValBeg + 1);
  if (quoteValBeg == std::string::npos || quoteValEnd == std::string::npos ||
      quoteValEnd <= quoteValBeg + 1) {
    return false;
  }
  const std::string descr =
      header.substr(quoteValBeg + 1, quoteValEnd - quoteValBeg - 1);
  if (descr.size() < 3) {
    return false;
  }
  out.typeChar = descr[1];
  out.wordSize = static_cast<size_t>(std::strtoul(descr.c_str() + 2, nullptr, 10));

  const auto fortranPos = header.find("fortran_order");
  if (fortranPos != std::string::npos) {
    const auto fortranStr = header.substr(fortranPos, header.size() - fortranPos);
    out.fortran = fortranStr.find("True") != std::string::npos ||
                  fortranStr.find("true") != std::string::npos;
  }

  const auto shapePos = header.find("shape");
  const auto lparen = header.find('(', shapePos);
  const auto rparen = header.find(')', lparen);
  if (lparen == std::string::npos || rparen == std::string::npos ||
      rparen <= lparen + 1) {
    return false;
  }

  std::string shapeStr = header.substr(lparen + 1, rparen - lparen - 1);
  std::stringstream ss(shapeStr);
  std::string token;
  out.shape.clear();
  while (std::getline(ss, token, ',')) {
    token = ltrim(token);
    if (token.empty()) {
      continue;
    }
    try {
      out.shape.push_back(static_cast<size_t>(std::stoull(token)));
    } catch (...) {
      return false;
    }
  }
  return !out.shape.empty();
}

bool loadNpyFromZip(mz_zip_archive& zip,
                    const std::string& key,
                    NpyArrayData& out) {
  std::string filename = key + ".npy";
  const int fileIndex =
      mz_zip_reader_locate_file(&zip, filename.c_str(), nullptr, 0);
  if (fileIndex < 0) {
    return false;
  }

  size_t uncompressedSize = 0;
  void* buffer =
      mz_zip_reader_extract_to_heap(&zip, fileIndex, &uncompressedSize, 0);
  if (!buffer || uncompressedSize < 10) {
    if (buffer) {
      mz_free(buffer);
    }
    return false;
  }

  out.data.assign(static_cast<unsigned char*>(buffer),
                  static_cast<unsigned char*>(buffer) + uncompressedSize);
  mz_free(buffer);

  const unsigned char* ptr = out.data.data();
  if (std::memcmp(ptr, "\x93NUMPY", 6) != 0) {
    return false;
  }

  const unsigned char verMajor = ptr[6];
  size_t headerLen = 0;
  size_t headerOffset = 0;
  if (verMajor == 1) {
    headerOffset = 10;
    headerLen = static_cast<size_t>(ptr[8]) |
                (static_cast<size_t>(ptr[9]) << 8);
  } else if (verMajor == 2) {
    headerOffset = 12;
    headerLen = static_cast<size_t>(ptr[8]) |
                (static_cast<size_t>(ptr[9]) << 8) |
                (static_cast<size_t>(ptr[10]) << 16) |
                (static_cast<size_t>(ptr[11]) << 24);
  } else {
    return false;
  }

  if (headerOffset + headerLen > out.data.size()) {
    return false;
  }

  const std::string header(reinterpret_cast<const char*>(ptr + headerOffset),
                           headerLen);
  if (!parseNpyHeader(header, out)) {
    return false;
  }

  const size_t dataOffset = headerOffset + headerLen;
  size_t expectedSize = out.wordSize;
  for (size_t dim : out.shape) {
    expectedSize *= dim;
  }

  if (dataOffset + expectedSize > out.data.size()) {
    return false;
  }

  std::vector<unsigned char> payload(out.data.begin() + dataOffset,
                                     out.data.begin() + dataOffset + expectedSize);
  out.data.swap(payload);
  return true;
}

template <typename T>
bool copyArray(const NpyArrayData& npy,
               std::vector<T>& out,
               char expectedType,
               size_t expectedWordSize) {
  if (npy.typeChar != expectedType || npy.wordSize != expectedWordSize) {
    return false;
  }
  const size_t count = npy.elementCount();
  if (npy.data.size() < count * expectedWordSize) {
    return false;
  }
  out.resize(count);
  std::memcpy(out.data(), npy.data.data(), count * expectedWordSize);
  return true;
}

bool loadScalarFloat(mz_zip_archive& zip,
                     const std::string& key,
                     float& out) {
  NpyArrayData npy;
  if (!loadNpyFromZip(zip, key, npy)) {
    return false;
  }
  if (npy.typeChar != 'f' || npy.wordSize != sizeof(float) ||
      npy.elementCount() != 1) {
    return false;
  }
  std::memcpy(&out, npy.data.data(), sizeof(float));
  return true;
}

bool loadScalarInt(mz_zip_archive& zip, const std::string& key, int& out) {
  NpyArrayData npy;
  if (!loadNpyFromZip(zip, key, npy)) {
    return false;
  }
  if ((npy.typeChar != 'i' && npy.typeChar != 'l' && npy.typeChar != 'q') ||
      npy.elementCount() != 1) {
    return false;
  }
  // Support 32-bit or 64-bit integer payloads
  if (npy.wordSize == sizeof(int)) {
    std::memcpy(&out, npy.data.data(), sizeof(int));
    return true;
  }
  if (npy.wordSize == sizeof(std::int64_t)) {
    std::int64_t tmp = 0;
    std::memcpy(&tmp, npy.data.data(), sizeof(std::int64_t));
    out = static_cast<int>(tmp);
    return true;
  }
  return false;
}

bool hasNpyKey(mz_zip_archive& zip, const std::string& key) {
  const std::string filename = key + ".npy";
  return mz_zip_reader_locate_file(&zip, filename.c_str(), nullptr, 0) >= 0;
}

int deduceShDegree(size_t shRestCount) {
  if (shRestCount == 0) {
    return 0;
  }
  const double coeffs = static_cast<double>(shRestCount) / 3.0 + 1.0;
  return std::max(0, static_cast<int>(std::round(std::sqrt(coeffs) - 1.0)));
}
}  // namespace

GaussianSplattingImporter::GaussianSplattingImporter() = default;

Mn::Trade::ImporterFeatures GaussianSplattingImporter::doFeatures() const {
  return Mn::Trade::ImporterFeature::OpenData |
         Mn::Trade::ImporterFeature::FileCallback;
}

bool GaussianSplattingImporter::doIsOpened() const {
  return opened_;
}

void GaussianSplattingImporter::doOpenFile(
    Corrade::Containers::StringView filename) {
  // Clear any previous data
  doClose();

  std::string filenameStr{filename};
  if (Cr::Utility::String::endsWith(filenameStr, ".npz")) {
    ESP_DEBUG() << "Opening Gaussian Splatting NPZ file:" << filenameStr;
    if (!loadFromNpz(filenameStr)) {
      ESP_ERROR() << "Failed to parse NPZ file:" << filenameStr;
      return;
    }
    opened_ = true;
    return;
  }

  std::ifstream file(filenameStr, std::ios::binary);

  if (!file.is_open()) {
    ESP_ERROR() << "Failed to open 3D/4DGS PLY file:" << filenameStr;
    return;
  }

  is4D_ = Cr::Utility::String::endsWith(filenameStr, ".4dgs.ply");
  ESP_DEBUG() << "Opening Gaussian Splatting PLY file:" << filenameStr;

  // Parse PLY header
  if (!parsePLYHeader(file)) {
    ESP_ERROR() << "Failed to parse PLY header from:" << filenameStr;
    file.close();
    return;
  }

  // Read binary data
  if (!readBinaryData(file)) {
    ESP_ERROR() << "Failed to read binary data from:" << filenameStr;
    file.close();
    return;
  }

  file.close();
  opened_ = true;

  ESP_DEBUG() << "Successfully loaded" << gaussianCount_
              << "Gaussians from PLY file"
              << (is4D_ ? " (4DGS detected)" : " (3DGS detected)");
}

void GaussianSplattingImporter::doClose() {
  opened_ = false;
  gaussianCount_ = 0;
  shRestCount_ = 0;
  bytesPerVertex_ = 0;
  properties_.clear();
  positions_.clear();
  normals_.clear();
  sh_dc_.clear();
  sh_rest_.clear();
  opacities_.clear();
  scales_.clear();
  rotations_.clear();
  rotationsR_.clear();
  times_.clear();
  timeScales_.clear();
  motions_.clear();
  motionCoeffs_.clear();
  hdbscanCache_.clear();
  duration_ = 0.0f;
  interval_ = 0.0f;
  foregroundInterval_ = 0.0f;
  is4D_ = false;
  hasMotion9_ = false;
  isHybrid_ = false;
  foregroundData_ = GaussianSetData{};
  backgroundData_ = GaussianSetData{};
}

Mn::UnsignedInt GaussianSplattingImporter::doMeshCount() const {
  return opened_ ? 1 : 0;
}

Corrade::Containers::Optional<Mn::Trade::MeshData>
GaussianSplattingImporter::doMesh(Mn::UnsignedInt id, Mn::UnsignedInt level) {
  if (!opened_ || id != 0) {
    ESP_ERROR() << "Invalid mesh request: opened=" << opened_ << ", id=" << id;
    return Corrade::Containers::NullOpt;
  }

  // Convert Gaussian positions to MeshData for compatibility
  // This creates a point cloud representation
  Corrade::Containers::Array<Magnum::Vector3> positionData(gaussianCount_);
  for (size_t i = 0; i < gaussianCount_; ++i) {
    positionData[i] = positions_[i];
  }

  // Create MeshData with positions only (as a point cloud)
  return Mn::Trade::MeshData{
      Mn::MeshPrimitive::Points,
      {},
      {Mn::Trade::MeshAttributeData{Mn::Trade::MeshAttribute::Position,
                                    Cr::Containers::arrayView(positionData)}},
      static_cast<Mn::UnsignedInt>(gaussianCount_)};
}

bool GaussianSplattingImporter::parsePLYHeader(std::ifstream& file) {
  std::string line;

  // Read magic number
  std::getline(file, line);
  if (line != "ply") {
    ESP_ERROR() << "Not a valid PLY file (missing 'ply' header)";
    return false;
  }

  // Read format
  std::getline(file, line);
  if (line.find("format binary_little_endian") == std::string::npos) {
    ESP_ERROR() << "Only binary_little_endian format is supported";
    return false;
  }

  // Parse header to find vertex count and properties
  bool foundVertexElement = false;
  bool parsingVertexProps = false;
  shRestCount_ = 0;
  bytesPerVertex_ = 0;
  properties_.clear();

  size_t offset = 0;

  while (std::getline(file, line)) {
    if (line == "end_header") {
      break;
    }

    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;

    if (keyword == "element") {
      std::string elementType;
      iss >> elementType;
      if (elementType == "vertex") {
        iss >> gaussianCount_;
        foundVertexElement = true;
        parsingVertexProps = true;
        offset = 0;
        ESP_DEBUG() << "Found" << gaussianCount_ << "vertices in PLY";
      } else {
        // Any new element terminates vertex property parsing
        parsingVertexProps = false;
      }
    } else if (keyword == "property" && parsingVertexProps) {
      std::string propType, propName;
      iss >> propType >> propName;

      PropertyInfo::Type type;
      size_t size = 0;
      if (propType == "float" || propType == "float32") {
        type = PropertyInfo::Type::Float;
        size = sizeof(float);
      } else if (propType == "uchar" || propType == "uint8") {
        type = PropertyInfo::Type::UChar;
        size = sizeof(unsigned char);
      } else {
        ESP_ERROR() << "Unsupported PLY property type:" << propType;
        return false;
      }

      PropertyInfo info;
      info.name = propName;
      info.type = type;
      info.offset = offset;
      info.size = size;
      properties_.push_back(info);
      offset += size;

      if (isGaussianRestProperty(propName)) {
        shRestCount_++;
      }

      if (propName == "t" || propName == "t_scale" ||
          propName.rfind("motion_", 0) == 0) {
        is4D_ = true;
      }
    }
  }

  bytesPerVertex_ = offset;

  if (!foundVertexElement || gaussianCount_ == 0 || bytesPerVertex_ == 0) {
    ESP_ERROR() << "Invalid PLY header: no vertex element found";
    return false;
  }

  ESP_DEBUG() << "PLY header parsed: " << gaussianCount_
              << " Gaussians with " << shRestCount_
              << " SH rest coefficients and" << properties_.size()
              << " properties (" << bytesPerVertex_ << " bytes per vertex)";

  return true;
}

bool GaussianSplattingImporter::loadFromNpz(const std::string& filename) {
  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  if (!mz_zip_reader_init_file(&zip, filename.c_str(), 0)) {
    ESP_ERROR() << "Failed to open NPZ file:" << filename;
    return false;
  }

  bool hasForeground = hasNpyKey(zip, "network.fg_fdgs._xyz");
  bool hasBackground = hasNpyKey(zip, "network.bg_fdgs._xyz");

  bool success = false;
  if (hasForeground && hasBackground) {
    success = loadInteractiveNpz(zip, filename);
  } else {
    success = loadTemporalNpz(zip, filename);
  }

  mz_zip_reader_end(&zip);
  if (success) {
    rebuildCombinedBuffers();
  }
  return success;
}

bool GaussianSplattingImporter::loadTemporalNpz(mz_zip_archive& zip,
                                                const std::string& filename) {
  foregroundData_ = GaussianSetData{};
  backgroundData_ = GaussianSetData{};
  isHybrid_ = false;
  duration_ = 0.0f;
  interval_ = 0.0f;
  foregroundInterval_ = 0.0f;
  loadScalarFloat(zip, "network.duration", duration_);
  loadScalarFloat(zip, "network.interval", interval_);
  loadScalarFloat(zip, "network.tfgs.fdgs.interval", foregroundInterval_);
  if (foregroundInterval_ <= 0.0f && interval_ > 0.0f) {
    foregroundInterval_ = interval_;
  }

  NpyArrayData xyzNpy;
  if (!loadNpyFromZip(zip, "network.tfgs.fdgs._xyz", xyzNpy) ||
      xyzNpy.shape.size() != 2 || xyzNpy.shape[1] != 3) {
    ESP_ERROR() << "Missing or invalid _xyz array in" << filename;
    return false;
  }

  foregroundData_.count = xyzNpy.shape[0];
  std::vector<float> xyzFlat;
  if (!copyArray(xyzNpy, xyzFlat, 'f', sizeof(float))) {
    ESP_ERROR() << "Unsupported dtype for _xyz in" << filename;
    return false;
  }

  auto loadVectorArray = [&](const std::string& key,
                             size_t width,
                             std::vector<float>& dst) -> bool {
    NpyArrayData npy;
    if (!loadNpyFromZip(zip, key, npy)) {
      return false;
    }
    if (npy.shape.size() != 2 || npy.shape[0] != foregroundData_.count ||
        npy.shape[1] != width) {
      ESP_ERROR() << "Unexpected shape for" << key << "in" << filename;
      return false;
    }
    if (!copyArray(npy, dst, 'f', sizeof(float))) {
      ESP_ERROR() << "Unsupported dtype for" << key << "in" << filename;
      return false;
    }
    return true;
  };

  auto loadDynamicArray = [&](const std::string& key,
                              std::vector<float>& dst,
                              size_t& widthOut) -> bool {
    NpyArrayData npy;
    if (!loadNpyFromZip(zip, key, npy)) {
      return false;
    }
    if (npy.shape.size() != 2 || npy.shape[0] != foregroundData_.count) {
      ESP_ERROR() << "Unexpected shape for" << key << "in" << filename;
      return false;
    }
    widthOut = npy.shape[1];
    if (!copyArray(npy, dst, 'f', sizeof(float))) {
      ESP_ERROR() << "Unsupported dtype for" << key << "in" << filename;
      return false;
    }
    return true;
  };

  std::vector<float> tFlat;
  std::vector<float> opacityFlat;
  std::vector<float> scaleFlat;
  std::vector<float> scaleTFlat;
  std::vector<float> rotLFlat;
  std::vector<float> rotRFlat;
  std::vector<float> motionFlat;
  std::vector<float> shDcFlat;
  std::vector<float> shRestFlat;
  std::vector<float> semanticFlat;
  size_t semanticWidth = 0;

  loadVectorArray("network.tfgs.fdgs._t", 1, tFlat);
  loadVectorArray("network.tfgs.fdgs._opacity", 1, opacityFlat);
  loadVectorArray("network.tfgs.fdgs._scaling", 3, scaleFlat);
  loadVectorArray("network.tfgs.fdgs._scaling_t", 1, scaleTFlat);
  loadVectorArray("network.tfgs.fdgs._rotation_l", 4, rotLFlat);
  loadVectorArray("network.tfgs.fdgs._rotation_r", 4, rotRFlat);
  size_t motionStrideDetected = 0;
  loadDynamicArray("network.tfgs.fdgs._motion", motionFlat,
                   motionStrideDetected);
  {
    NpyArrayData npy;
    if (loadNpyFromZip(zip, "network.tfgs.fdgs._semantic", npy) &&
        npy.shape.size() == 2 && npy.shape[0] == foregroundData_.count) {
      semanticWidth = npy.shape[1];
      if (!copyArray(npy, semanticFlat, 'f', sizeof(float))) {
        semanticFlat.clear();
        semanticWidth = 0;
        ESP_WARNING() << "Unsupported dtype for tfgs semantics in" << filename;
      }
    }
  }

  // SH DC: (N, 1, 3)
  {
    NpyArrayData npy;
    if (loadNpyFromZip(zip, "network.tfgs.fdgs._features_dc", npy) &&
        npy.shape.size() == 3 && npy.shape[0] == foregroundData_.count &&
        npy.shape[2] == 3) {
      if (!copyArray(npy, shDcFlat, 'f', sizeof(float))) {
        ESP_WARNING() << "Unsupported dtype for _features_dc in" << filename;
        shDcFlat.clear();
      }
    }
  }

  // SH rest: (N, M, 3)
  size_t shRestBlocks = 0;
  {
    NpyArrayData npy;
    if (loadNpyFromZip(zip, "network.tfgs.fdgs._features_rest", npy) &&
        npy.shape.size() == 3 && npy.shape[0] == foregroundData_.count &&
        npy.shape[2] == 3) {
      shRestBlocks = npy.shape[1];
      if (!copyArray(npy, shRestFlat, 'f', sizeof(float))) {
        ESP_WARNING() << "Unsupported dtype for _features_rest in" << filename;
        shRestFlat.clear();
        shRestBlocks = 0;
      }
    }
  }

  // Scalar metadata (optional)
  float sceneScale = 1.0f;
  loadScalarFloat(zip, "network.scene_scale", sceneScale);
  float timeDilation = 1.0f;
  loadScalarFloat(zip, "network.time_dilation", timeDilation);

  int shDegree = 0;
  loadScalarInt(zip, "network.tfgs.active_sh_degree", shDegree);
  int shDegreeT = 0;
  loadScalarInt(zip, "network.tfgs.active_sh_degree_t", shDegreeT);

  foregroundData_.shRestCount = shRestBlocks * 3;
  foregroundData_.activeShDegree =
      shDegree > 0 ? shDegree : deduceShDegree(foregroundData_.shRestCount);
  foregroundData_.activeShDegreeT = shDegreeT;
  foregroundData_.hasRotationR = !rotRFlat.empty();
  const size_t motionStride =
      motionStrideDetected > 0 ? motionStrideDetected : (motionFlat.empty() ? 3 : 9);
  foregroundData_.motionStride = static_cast<int>(motionStride);
  foregroundData_.hasMotion9 = motionStride >= 9;
  foregroundData_.semanticDim = semanticWidth;

  foregroundData_.positions.reserve(foregroundData_.count);
  foregroundData_.normals.assign(foregroundData_.count, Mn::Vector3{0.0f});
  foregroundData_.sh_dc.reserve(foregroundData_.count);
  foregroundData_.sh_rest.reserve(foregroundData_.count);
  foregroundData_.opacities.reserve(foregroundData_.count);
  foregroundData_.scales.reserve(foregroundData_.count);
  foregroundData_.rotations.reserve(foregroundData_.count);
  foregroundData_.rotationsR.reserve(foregroundData_.count);
  foregroundData_.times.reserve(foregroundData_.count);
  foregroundData_.timeScales.reserve(foregroundData_.count);
  foregroundData_.motions.reserve(foregroundData_.count);
  foregroundData_.motionCoeffs.reserve(foregroundData_.count);
  foregroundData_.semantics.assign(foregroundData_.count,
                                   std::vector<float>(semanticWidth, 0.0f));

  const float scaleAdd = sceneScale > 0.0f ? std::log(sceneScale) : 0.0f;
  const bool hasScale = !scaleFlat.empty();
  const bool hasRotR = !rotRFlat.empty();
  const bool hasScaleT = !scaleTFlat.empty();
  const bool hasTime = !tFlat.empty();
  const bool hasOpacity = !opacityFlat.empty();
  const bool hasRotL = !rotLFlat.empty();
  for (size_t i = 0; i < foregroundData_.count; ++i) {
    const size_t base3 = i * 3;
    const size_t base4 = i * 4;

    foregroundData_.positions.emplace_back(
        sceneScale * xyzFlat[base3 + 0],
        sceneScale * xyzFlat[base3 + 1],
        sceneScale * xyzFlat[base3 + 2]);
    const bool hasDc = !shDcFlat.empty() && shDcFlat.size() >= (i * 3 + 3);
    foregroundData_.sh_dc.emplace_back(
        hasDc ? Mn::Vector3{shDcFlat[i * 3 + 0], shDcFlat[i * 3 + 1],
                            shDcFlat[i * 3 + 2]}
              : Mn::Vector3{0.0f});

    foregroundData_.opacities.push_back(hasOpacity ? opacityFlat[i] : 1.0f);
    const float sx = hasScale ? scaleFlat[base3 + 0] : 0.0f;
    const float sy = hasScale ? scaleFlat[base3 + 1] : 0.0f;
    const float sz = hasScale ? scaleFlat[base3 + 2] : 0.0f;
    foregroundData_.scales.emplace_back(sx + scaleAdd, sy + scaleAdd,
                                        sz + scaleAdd);

    if (hasRotL) {
      foregroundData_.rotations.emplace_back(
          Mn::Vector3{rotLFlat[base4 + 1], rotLFlat[base4 + 2],
                      rotLFlat[base4 + 3]},
          rotLFlat[base4 + 0]);
    } else {
      foregroundData_.rotations.emplace_back(Mn::Vector3{0.0f}, 1.0f);
    }

    if (hasRotR) {
      foregroundData_.rotationsR.emplace_back(
          Mn::Vector3{rotRFlat[base4 + 1], rotRFlat[base4 + 2],
                      rotRFlat[base4 + 3]},
          rotRFlat[base4 + 0]);
    } else {
      foregroundData_.rotationsR.emplace_back(Mn::Vector3{0.0f}, 1.0f);
    }

    foregroundData_.times.push_back(hasTime ? tFlat[i] * timeDilation : 0.0f);
    foregroundData_.timeScales.push_back(hasScaleT ? scaleTFlat[i] : 0.0f);

    Mn::Vector3 motion{0.0f};
    std::array<float, 9> coeffs{};
    if (!motionFlat.empty()) {
      const size_t baseMotion = i * motionStride;
      motion = Mn::Vector3{motionFlat[baseMotion + 0],
                           motionFlat[baseMotion + 1],
                           motionFlat[baseMotion + 2]} *
               sceneScale;
      for (size_t k = 0; k < std::min<size_t>(motionStride, 9); ++k) {
        coeffs[k] = motionFlat[baseMotion + k] * sceneScale;
      }
    }
    foregroundData_.motions.push_back(motion);
    foregroundData_.motionCoeffs.push_back(coeffs);

    std::vector<float> rest(foregroundData_.shRestCount, 0.0f);
    if (!shRestFlat.empty() && shRestBlocks > 0) {
      const size_t base = i * shRestBlocks * 3;
      for (size_t block = 0; block < shRestBlocks; ++block) {
        for (size_t c = 0; c < 3; ++c) {
          const size_t src = base + block * 3 + c;
          const size_t dst = c * shRestBlocks + block;
          if (src < shRestFlat.size() && dst < rest.size()) {
            rest[dst] = shRestFlat[src];
          }
        }
      }
    }
    foregroundData_.sh_rest.push_back(std::move(rest));
    if (!semanticFlat.empty() && semanticWidth > 0) {
      auto& dst = foregroundData_.semantics[i];
      const size_t base = i * semanticWidth;
      for (size_t j = 0; j < semanticWidth && base + j < semanticFlat.size(); ++j) {
        dst[j] = semanticFlat[base + j];
      }
    }
  }

  is4D_ = true;
  {
    NpyArrayData npy;
    if (loadNpyFromZip(zip, "network.hdbscan_cache", npy) &&
        npy.shape.size() == 2 && !foregroundData_.semantics.empty()) {
      const size_t dim = npy.shape[1];
      std::vector<float> cacheFlat;
      if (copyArray(npy, cacheFlat, 'f', sizeof(float))) {
        const size_t clusters = npy.shape[0];
        hdbscanCache_.assign(clusters, std::vector<float>(dim, 0.0f));
        for (size_t c = 0; c < clusters; ++c) {
          const size_t base = c * dim;
          for (size_t j = 0; j < dim && base + j < cacheFlat.size(); ++j) {
            hdbscanCache_[c][j] = cacheFlat[base + j];
          }
        }
      }
    }
  }
  ESP_DEBUG() << "Parsed InteractiveStreet NPZ with" << foregroundData_.count
              << "foreground and" << backgroundData_.count
              << "background Gaussians.";
  return true;
}

bool GaussianSplattingImporter::loadInteractiveNpz(mz_zip_archive& zip,
                                                   const std::string& filename) {
  foregroundData_ = GaussianSetData{};
  backgroundData_ = GaussianSetData{};
  isHybrid_ = true;
  duration_ = 0.0f;
  interval_ = 0.0f;
  foregroundInterval_ = 0.0f;

  loadScalarFloat(zip, "network.duration", duration_);
  loadScalarFloat(zip, "network.interval", interval_);
  loadScalarFloat(zip, "network.fg_fdgs.interval", foregroundInterval_);
  if (foregroundInterval_ <= 0.0f && interval_ > 0.0f) {
    foregroundInterval_ = interval_;
  }

  float sceneScale = 1.0f;
  loadScalarFloat(zip, "network.scene_scale", sceneScale);
  float timeDilation = 1.0f;
  loadScalarFloat(zip, "network.time_dilation", timeDilation);

  // ---------------- Foreground (4D) ----------------
  {
    NpyArrayData xyzNpy;
    if (!loadNpyFromZip(zip, "network.fg_fdgs._xyz", xyzNpy) ||
        xyzNpy.shape.size() != 2 || xyzNpy.shape[1] != 3) {
      ESP_ERROR() << "Missing foreground _xyz in" << filename;
      return false;
    }
    foregroundData_.count = xyzNpy.shape[0];

    std::vector<float> xyzFlat;
    if (!copyArray(xyzNpy, xyzFlat, 'f', sizeof(float))) {
      ESP_ERROR() << "Unsupported dtype for fg _xyz in" << filename;
      return false;
    }

    auto loadVectorArray = [&](const std::string& key,
                               size_t width,
                               std::vector<float>& dst) -> bool {
      NpyArrayData npy;
      if (!loadNpyFromZip(zip, key, npy)) {
        return false;
      }
      if (npy.shape.size() != 2 || npy.shape[0] != foregroundData_.count ||
          npy.shape[1] != width) {
        ESP_ERROR() << "Unexpected shape for" << key << "in" << filename;
        return false;
      }
      if (!copyArray(npy, dst, 'f', sizeof(float))) {
        ESP_ERROR() << "Unsupported dtype for" << key << "in" << filename;
        return false;
      }
      return true;
    };

    auto loadDynamicArray = [&](const std::string& key,
                                std::vector<float>& dst,
                                size_t& widthOut) -> bool {
      NpyArrayData npy;
      if (!loadNpyFromZip(zip, key, npy)) {
        return false;
      }
      if (npy.shape.size() != 2 || npy.shape[0] != foregroundData_.count) {
        ESP_ERROR() << "Unexpected shape for" << key << "in" << filename;
        return false;
      }
      widthOut = npy.shape[1];
      if (!copyArray(npy, dst, 'f', sizeof(float))) {
        ESP_ERROR() << "Unsupported dtype for" << key << "in" << filename;
        return false;
      }
      return true;
    };

    std::vector<float> tFlat;
    std::vector<float> opacityFlat;
    std::vector<float> scaleFlat;
    std::vector<float> scaleTFlat;
    std::vector<float> rotLFlat;
    std::vector<float> rotRFlat;
    std::vector<float> motionFlat;
    std::vector<float> shDcFlat;
    std::vector<float> shRestFlat;
    std::vector<float> semanticFlat;
    size_t semanticWidth = 0;

    loadVectorArray("network.fg_fdgs._t", 1, tFlat);
    loadVectorArray("network.fg_fdgs._opacity", 1, opacityFlat);
    loadVectorArray("network.fg_fdgs._scaling", 3, scaleFlat);
    loadVectorArray("network.fg_fdgs._scaling_t", 1, scaleTFlat);
    loadVectorArray("network.fg_fdgs._rotation_l", 4, rotLFlat);
    loadVectorArray("network.fg_fdgs._rotation_r", 4, rotRFlat);
    size_t motionStrideDetected = 0;
    loadDynamicArray("network.fg_fdgs._motion", motionFlat,
                     motionStrideDetected);
    {
      NpyArrayData npy;
      if (loadNpyFromZip(zip, "network.fg_fdgs._semantic", npy) &&
          npy.shape.size() == 2 && npy.shape[0] == foregroundData_.count) {
        semanticWidth = npy.shape[1];
        if (!copyArray(npy, semanticFlat, 'f', sizeof(float))) {
          semanticFlat.clear();
          semanticWidth = 0;
          ESP_WARNING() << "Unsupported dtype for fg semantics in" << filename;
        }
      }
    }

    {
      NpyArrayData npy;
      if (loadNpyFromZip(zip, "network.fg_fdgs._features_dc", npy) &&
          npy.shape.size() == 3 && npy.shape[0] == foregroundData_.count &&
          npy.shape[2] == 3) {
        if (!copyArray(npy, shDcFlat, 'f', sizeof(float))) {
          ESP_WARNING() << "Unsupported dtype for fg _features_dc in"
                        << filename;
          shDcFlat.clear();
        }
      }
    }

    size_t shRestBlocks = 0;
    {
      NpyArrayData npy;
      if (loadNpyFromZip(zip, "network.fg_fdgs._features_rest", npy) &&
          npy.shape.size() == 3 && npy.shape[0] == foregroundData_.count &&
          npy.shape[2] == 3) {
        shRestBlocks = npy.shape[1];
        if (!copyArray(npy, shRestFlat, 'f', sizeof(float))) {
          ESP_WARNING() << "Unsupported dtype for fg _features_rest in"
                        << filename;
          shRestFlat.clear();
          shRestBlocks = 0;
        }
      }
    }

    int shDegree = 0;
    loadScalarInt(zip, "network.fg_fdgs.active_sh_degree", shDegree);
    int shDegreeT = 0;
    loadScalarInt(zip, "network.fg_fdgs.active_sh_degree_t", shDegreeT);

    foregroundData_.shRestCount = shRestBlocks * 3;
    foregroundData_.activeShDegree =
        shDegree > 0 ? shDegree : deduceShDegree(foregroundData_.shRestCount);
    foregroundData_.activeShDegreeT = shDegreeT;
    foregroundData_.hasRotationR = !rotRFlat.empty();
    const size_t motionStride =
        motionStrideDetected > 0
            ? motionStrideDetected
            : (motionFlat.empty() ? 3 : 9);
    foregroundData_.motionStride = static_cast<int>(motionStride);
    foregroundData_.hasMotion9 = motionStride >= 9;
    foregroundData_.semanticDim = semanticWidth;

    foregroundData_.positions.reserve(foregroundData_.count);
    foregroundData_.normals.assign(foregroundData_.count, Mn::Vector3{0.0f});
    foregroundData_.sh_dc.reserve(foregroundData_.count);
    foregroundData_.sh_rest.reserve(foregroundData_.count);
    foregroundData_.opacities.reserve(foregroundData_.count);
    foregroundData_.scales.reserve(foregroundData_.count);
    foregroundData_.rotations.reserve(foregroundData_.count);
    foregroundData_.rotationsR.reserve(foregroundData_.count);
    foregroundData_.times.reserve(foregroundData_.count);
    foregroundData_.timeScales.reserve(foregroundData_.count);
    foregroundData_.motions.reserve(foregroundData_.count);
    foregroundData_.motionCoeffs.reserve(foregroundData_.count);
    foregroundData_.semantics.assign(foregroundData_.count,
                                     std::vector<float>(semanticWidth, 0.0f));

    const float scaleAdd = sceneScale > 0.0f ? std::log(sceneScale) : 0.0f;
    const bool hasScale = !scaleFlat.empty();
    const bool hasRotR = !rotRFlat.empty();
    const bool hasScaleT = !scaleTFlat.empty();
    const bool hasTime = !tFlat.empty();
    const bool hasOpacity = !opacityFlat.empty();
    const bool hasRotL = !rotLFlat.empty();
    for (size_t i = 0; i < foregroundData_.count; ++i) {
      const size_t base3 = i * 3;
      const size_t base4 = i * 4;

      foregroundData_.positions.emplace_back(
          sceneScale * xyzFlat[base3 + 0],
          sceneScale * xyzFlat[base3 + 1],
          sceneScale * xyzFlat[base3 + 2]);
      const bool hasDc = !shDcFlat.empty() && shDcFlat.size() >= (i * 3 + 3);
      foregroundData_.sh_dc.emplace_back(
          hasDc ? Mn::Vector3{shDcFlat[i * 3 + 0], shDcFlat[i * 3 + 1],
                              shDcFlat[i * 3 + 2]}
                : Mn::Vector3{0.0f});

      foregroundData_.opacities.push_back(hasOpacity ? opacityFlat[i] : 1.0f);
      const float sx = hasScale ? scaleFlat[base3 + 0] : 0.0f;
      const float sy = hasScale ? scaleFlat[base3 + 1] : 0.0f;
      const float sz = hasScale ? scaleFlat[base3 + 2] : 0.0f;
      foregroundData_.scales.emplace_back(sx + scaleAdd, sy + scaleAdd,
                                          sz + scaleAdd);

      if (hasRotL) {
        foregroundData_.rotations.emplace_back(
            Mn::Vector3{rotLFlat[base4 + 1], rotLFlat[base4 + 2],
                        rotLFlat[base4 + 3]},
            rotLFlat[base4 + 0]);
      } else {
        foregroundData_.rotations.emplace_back(Mn::Vector3{0.0f}, 1.0f);
      }

      if (hasRotR) {
        foregroundData_.rotationsR.emplace_back(
            Mn::Vector3{rotRFlat[base4 + 1], rotRFlat[base4 + 2],
                        rotRFlat[base4 + 3]},
            rotRFlat[base4 + 0]);
      } else {
        foregroundData_.rotationsR.emplace_back(Mn::Vector3{0.0f}, 1.0f);
      }

      foregroundData_.times.push_back(hasTime ? tFlat[i] * timeDilation
                                              : 0.0f);
      foregroundData_.timeScales.push_back(hasScaleT ? scaleTFlat[i] : 0.0f);

      Mn::Vector3 motion{0.0f};
      std::array<float, 9> coeffs{};
      if (!motionFlat.empty()) {
        const size_t baseMotion = i * motionStride;
        motion = Mn::Vector3{motionFlat[baseMotion + 0],
                             motionFlat[baseMotion + 1],
                             motionFlat[baseMotion + 2]} *
                 sceneScale;
        for (size_t k = 0; k < std::min<size_t>(motionStride, 9); ++k) {
          coeffs[k] = motionFlat[baseMotion + k] * sceneScale;
        }
      }
      foregroundData_.motions.push_back(motion);
      foregroundData_.motionCoeffs.push_back(coeffs);

      std::vector<float> rest(foregroundData_.shRestCount, 0.0f);
      if (!shRestFlat.empty() && shRestBlocks > 0) {
        const size_t base = i * shRestBlocks * 3;
        for (size_t block = 0; block < shRestBlocks; ++block) {
          for (size_t c = 0; c < 3; ++c) {
            const size_t src = base + block * 3 + c;
            const size_t dst = c * shRestBlocks + block;
            if (src < shRestFlat.size() && dst < rest.size()) {
              rest[dst] = shRestFlat[src];
            }
          }
        }
      }
      foregroundData_.sh_rest.push_back(std::move(rest));
      if (!semanticFlat.empty() && semanticWidth > 0) {
        auto& dst = foregroundData_.semantics[i];
        const size_t base = i * semanticWidth;
        for (size_t j = 0; j < semanticWidth && base + j < semanticFlat.size();
             ++j) {
          dst[j] = semanticFlat[base + j];
        }
      }
    }
  }

  // ---------------- Background (3D) ----------------
  {
    NpyArrayData xyzNpy;
    if (!loadNpyFromZip(zip, "network.bg_fdgs._xyz", xyzNpy) ||
        xyzNpy.shape.size() != 2 || xyzNpy.shape[1] != 3) {
      ESP_ERROR() << "Missing background _xyz in" << filename;
      return false;
    }
    backgroundData_.count = xyzNpy.shape[0];

    std::vector<float> xyzFlat;
    if (!copyArray(xyzNpy, xyzFlat, 'f', sizeof(float))) {
      ESP_ERROR() << "Unsupported dtype for bg _xyz in" << filename;
      return false;
    }

    auto loadVectorArray = [&](const std::string& key,
                               size_t width,
                               std::vector<float>& dst) -> bool {
      NpyArrayData npy;
      if (!loadNpyFromZip(zip, key, npy)) {
        return false;
      }
      if (npy.shape.size() != 2 || npy.shape[0] != backgroundData_.count ||
          npy.shape[1] != width) {
        ESP_ERROR() << "Unexpected shape for" << key << "in" << filename;
        return false;
      }
      if (!copyArray(npy, dst, 'f', sizeof(float))) {
        ESP_ERROR() << "Unsupported dtype for" << key << "in" << filename;
        return false;
      }
      return true;
    };

    std::vector<float> opacityFlat;
    std::vector<float> scaleFlat;
    std::vector<float> rotFlat;
    std::vector<float> shDcFlat;
    std::vector<float> shRestFlat;

    loadVectorArray("network.bg_fdgs._opacity", 1, opacityFlat);
    loadVectorArray("network.bg_fdgs._scaling", 3, scaleFlat);
    loadVectorArray("network.bg_fdgs._rotation", 4, rotFlat);

    {
      NpyArrayData npy;
      if (loadNpyFromZip(zip, "network.bg_fdgs._features_dc", npy) &&
          npy.shape.size() == 3 && npy.shape[0] == backgroundData_.count &&
          npy.shape[2] == 3) {
        if (!copyArray(npy, shDcFlat, 'f', sizeof(float))) {
          ESP_WARNING() << "Unsupported dtype for bg _features_dc in"
                        << filename;
          shDcFlat.clear();
        }
      }
    }

    size_t shRestBlocks = 0;
    {
      NpyArrayData npy;
      if (loadNpyFromZip(zip, "network.bg_fdgs._features_rest", npy) &&
          npy.shape.size() == 3 && npy.shape[0] == backgroundData_.count &&
          npy.shape[2] == 3) {
        shRestBlocks = npy.shape[1];
        if (!copyArray(npy, shRestFlat, 'f', sizeof(float))) {
          ESP_WARNING() << "Unsupported dtype for bg _features_rest in"
                        << filename;
          shRestFlat.clear();
          shRestBlocks = 0;
        }
      }
    }

    int shDegree = 0;
    loadScalarInt(zip, "network.bg_fdgs.active_sh_degree", shDegree);
    backgroundData_.shRestCount = shRestBlocks * 3;
    backgroundData_.activeShDegree =
        shDegree > 0 ? shDegree : deduceShDegree(backgroundData_.shRestCount);
    backgroundData_.activeShDegreeT = 0;
    backgroundData_.hasRotationR = false;
    backgroundData_.motionStride = 3;
    backgroundData_.hasMotion9 = false;
    backgroundData_.semanticDim = 0;

    backgroundData_.positions.reserve(backgroundData_.count);
    backgroundData_.normals.assign(backgroundData_.count, Mn::Vector3{0.0f});
    backgroundData_.sh_dc.reserve(backgroundData_.count);
    backgroundData_.sh_rest.reserve(backgroundData_.count);
    backgroundData_.opacities.reserve(backgroundData_.count);
    backgroundData_.scales.reserve(backgroundData_.count);
    backgroundData_.rotations.reserve(backgroundData_.count);
    backgroundData_.rotationsR.assign(backgroundData_.count,
                                      Mn::Quaternion{Mn::Vector3{0.0f}, 1.0f});
    backgroundData_.times.assign(backgroundData_.count, 0.0f);
    backgroundData_.timeScales.assign(backgroundData_.count, 0.0f);
    backgroundData_.motions.assign(backgroundData_.count, Mn::Vector3{0.0f});
    backgroundData_.motionCoeffs.assign(backgroundData_.count, std::array<float, 9>{});
    backgroundData_.semantics.clear();

    const float scaleAdd = sceneScale > 0.0f ? std::log(sceneScale) : 0.0f;
    const bool hasScale = !scaleFlat.empty();
    const bool hasOpacity = !opacityFlat.empty();
    const bool hasRot = !rotFlat.empty();
    for (size_t i = 0; i < backgroundData_.count; ++i) {
      const size_t base3 = i * 3;
      const size_t base4 = i * 4;

      backgroundData_.positions.emplace_back(
          sceneScale * xyzFlat[base3 + 0],
          sceneScale * xyzFlat[base3 + 1],
          sceneScale * xyzFlat[base3 + 2]);
      const bool hasDc = !shDcFlat.empty() && shDcFlat.size() >= (i * 3 + 3);
      backgroundData_.sh_dc.emplace_back(
          hasDc ? Mn::Vector3{shDcFlat[i * 3 + 0], shDcFlat[i * 3 + 1],
                              shDcFlat[i * 3 + 2]}
                : Mn::Vector3{0.0f});

      backgroundData_.opacities.push_back(hasOpacity ? opacityFlat[i] : 1.0f);
      const float sx = hasScale ? scaleFlat[base3 + 0] : 0.0f;
      const float sy = hasScale ? scaleFlat[base3 + 1] : 0.0f;
      const float sz = hasScale ? scaleFlat[base3 + 2] : 0.0f;
      backgroundData_.scales.emplace_back(sx + scaleAdd, sy + scaleAdd,
                                          sz + scaleAdd);

      if (hasRot) {
        backgroundData_.rotations.emplace_back(
            Mn::Vector3{rotFlat[base4 + 1], rotFlat[base4 + 2],
                        rotFlat[base4 + 3]},
            rotFlat[base4 + 0]);
      } else {
        backgroundData_.rotations.emplace_back(Mn::Vector3{0.0f}, 1.0f);
      }

      std::vector<float> rest(backgroundData_.shRestCount, 0.0f);
      if (!shRestFlat.empty() && shRestBlocks > 0) {
        const size_t base = i * shRestBlocks * 3;
        for (size_t block = 0; block < shRestBlocks; ++block) {
          for (size_t c = 0; c < 3; ++c) {
            const size_t src = base + block * 3 + c;
            const size_t dst = c * shRestBlocks + block;
            if (src < shRestFlat.size() && dst < rest.size()) {
              rest[dst] = shRestFlat[src];
            }
          }
        }
      }
      backgroundData_.sh_rest.push_back(std::move(rest));
    }
  }

  {
    NpyArrayData npy;
    if (loadNpyFromZip(zip, "network.hdbscan_cache", npy) &&
        npy.shape.size() == 2 && !foregroundData_.semantics.empty()) {
      const size_t dim = npy.shape[1];
      std::vector<float> cacheFlat;
      if (copyArray(npy, cacheFlat, 'f', sizeof(float))) {
        const size_t clusters = npy.shape[0];
        hdbscanCache_.assign(clusters, std::vector<float>(dim, 0.0f));
        for (size_t c = 0; c < clusters; ++c) {
          const size_t base = c * dim;
          for (size_t j = 0; j < dim && base + j < cacheFlat.size(); ++j) {
            hdbscanCache_[c][j] = cacheFlat[base + j];
          }
        }
      }
    }
  }
  is4D_ = true;
  return true;
}

void GaussianSplattingImporter::rebuildCombinedBuffers() {
  gaussianCount_ = foregroundData_.count + backgroundData_.count;
  shRestCount_ =
      std::max(foregroundData_.shRestCount, backgroundData_.shRestCount);
  hasMotion9_ = foregroundData_.hasMotion9 || backgroundData_.hasMotion9;
  isHybrid_ =
      isHybrid_ || (foregroundData_.count > 0 && backgroundData_.count > 0);
  is4D_ = foregroundData_.count > 0 || is4D_;

  positions_.clear();
  normals_.clear();
  sh_dc_.clear();
  sh_rest_.clear();
  opacities_.clear();
  scales_.clear();
  rotations_.clear();
  rotationsR_.clear();
  times_.clear();
  timeScales_.clear();
  motions_.clear();
  motionCoeffs_.clear();

  positions_.reserve(gaussianCount_);
  normals_.reserve(gaussianCount_);
  sh_dc_.reserve(gaussianCount_);
  sh_rest_.reserve(gaussianCount_);
  opacities_.reserve(gaussianCount_);
  scales_.reserve(gaussianCount_);
  rotations_.reserve(gaussianCount_);
  rotationsR_.reserve(gaussianCount_);
  times_.reserve(gaussianCount_);
  timeScales_.reserve(gaussianCount_);
  motions_.reserve(gaussianCount_);
  motionCoeffs_.reserve(gaussianCount_);

  auto appendSet = [&](const GaussianSetData& set, bool isForeground) {
    for (size_t i = 0; i < set.count; ++i) {
      positions_.push_back(i < set.positions.size() ? set.positions[i]
                                                    : Mn::Vector3{0.0f});
      normals_.push_back(i < set.normals.size() ? set.normals[i]
                                                : Mn::Vector3{0.0f});
      sh_dc_.push_back(i < set.sh_dc.size() ? set.sh_dc[i]
                                            : Mn::Vector3{0.0f});
      std::vector<float> rest(shRestCount_, 0.0f);
      if (i < set.sh_rest.size()) {
        const auto& src = set.sh_rest[i];
        for (size_t j = 0; j < std::min(rest.size(), src.size()); ++j) {
          rest[j] = src[j];
        }
      }
      sh_rest_.push_back(std::move(rest));

      opacities_.push_back(i < set.opacities.size() ? set.opacities[i] : 1.0f);
      scales_.push_back(i < set.scales.size() ? set.scales[i]
                                              : Mn::Vector3{0.0f});
      rotations_.push_back(i < set.rotations.size()
                               ? set.rotations[i]
                               : Mn::Quaternion{Mn::Vector3{0.0f}, 1.0f});
      rotationsR_.push_back(set.hasRotationR && i < set.rotationsR.size()
                                ? set.rotationsR[i]
                                : Mn::Quaternion{Mn::Vector3{0.0f}, 1.0f});

      times_.push_back(i < set.times.size() ? set.times[i] : 0.0f);
      timeScales_.push_back(i < set.timeScales.size() ? set.timeScales[i]
                                                      : 0.0f);
      motions_.push_back(i < set.motions.size() ? set.motions[i]
                                                : Mn::Vector3{0.0f});
      std::array<float, 9> coeffs{};
      if (i < set.motionCoeffs.size()) {
        coeffs = set.motionCoeffs[i];
      }
      motionCoeffs_.push_back(coeffs);
    }
  };

  appendSet(foregroundData_, true);
  appendSet(backgroundData_, false);
}

float GaussianSplattingImporter::bytesToFloat(
    const unsigned char* bytes) const {
  float value;
  std::memcpy(&value, bytes, sizeof(float));
  return value;
}

bool GaussianSplattingImporter::readBinaryData(std::ifstream& file) {
  if (bytesPerVertex_ == 0 || properties_.empty()) {
    ESP_ERROR() << "No property layout available for PLY parsing.";
    return false;
  }

  GaussianSetData set;
  set.count = gaussianCount_;
  set.shRestCount = shRestCount_;
  set.activeShDegree = deduceShDegree(shRestCount_);
  set.activeShDegreeT = 0;
  set.hasRotationR = false;
  set.motionStride = 3;
  set.hasMotion9 = false;
  set.positions.reserve(gaussianCount_);
  set.normals.reserve(gaussianCount_);
  set.sh_dc.reserve(gaussianCount_);
  set.sh_rest.reserve(gaussianCount_);
  set.opacities.reserve(gaussianCount_);
  set.scales.reserve(gaussianCount_);
  set.rotations.reserve(gaussianCount_);
  set.rotationsR.reserve(gaussianCount_);
  set.times.reserve(gaussianCount_);
  set.timeScales.reserve(gaussianCount_);
  set.motions.reserve(gaussianCount_);
  set.motionCoeffs.reserve(gaussianCount_);

  std::vector<unsigned char> buffer(bytesPerVertex_);

  for (size_t i = 0; i < gaussianCount_; ++i) {
    file.read(reinterpret_cast<char*>(buffer.data()), bytesPerVertex_);

    if (!file) {
      ESP_ERROR() << "Failed to read Gaussian" << i << "of" << gaussianCount_;
      return false;
    }

    float x = 0.0f, y = 0.0f, z = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    Mn::Vector3 fdc{0.0f};
    bool hasFdc = false;
    std::vector<float> fRest(shRestCount_, 0.0f);
    float opacity = 1.0f;
    Mn::Vector3 scale{0.0f};
    float rot[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float time = 0.0f;
    float timeScale = 0.0f;
    Mn::Vector3 motion{0.0f};
    bool hasTemporal = false;
    bool hasColor = false;
    float color[3] = {0.0f, 0.0f, 0.0f};

    for (const auto& prop : properties_) {
      const unsigned char* ptr = buffer.data() + prop.offset;
      float value = 0.0f;
      if (prop.type == PropertyInfo::Type::Float) {
        value = bytesToFloat(ptr);
      } else {
        value = static_cast<float>(*ptr);
      }

      const std::string& name = prop.name;
      if (name == "x") {
        x = value;
      } else if (name == "y") {
        y = value;
      } else if (name == "z") {
        z = value;
      } else if (name == "nx" || name == "normal_x") {
        nx = value;
      } else if (name == "ny" || name == "normal_y") {
        ny = value;
      } else if (name == "nz" || name == "normal_z") {
        nz = value;
      } else if (name == "red") {
        color[0] = value;
        hasColor = true;
      } else if (name == "green") {
        color[1] = value;
        hasColor = true;
      } else if (name == "blue") {
        color[2] = value;
        hasColor = true;
      } else if (name.rfind("f_dc_", 0) == 0) {
        int idx = std::atoi(name.substr(5).c_str());
        if (idx >= 0 && idx < 3) {
          fdc[idx] = value;
          hasFdc = true;
        }
      } else if (isGaussianRestProperty(name)) {
        int idx = std::atoi(name.substr(7).c_str());
        if (idx >= 0 && static_cast<size_t>(idx) < fRest.size()) {
          fRest[idx] = value;
        }
      } else if (name == "opacity") {
        opacity = value;
      } else if (name == "scale_0") {
        scale[0] = value;
      } else if (name == "scale_1") {
        scale[1] = value;
      } else if (name == "scale_2") {
        scale[2] = value;
      } else if (name == "rot_0") {
        rot[0] = value;
      } else if (name == "rot_1") {
        rot[1] = value;
      } else if (name == "rot_2") {
        rot[2] = value;
      } else if (name == "rot_3") {
        rot[3] = value;
      } else if (name == "t") {
        time = value;
        hasTemporal = true;
      } else if (name == "t_scale") {
        timeScale = value;
        hasTemporal = true;
      } else if (name.rfind("motion_", 0) == 0) {
        int idx = std::atoi(name.substr(7).c_str());
        if (idx >= 0 && idx < 3) {
          motion[idx] = value;
          hasTemporal = true;
        }
      }
    }

    if (hasColor && !hasFdc) {
      // Fallback to RGB if SH DC coefficients were not provided explicitly.
      fdc = Mn::Vector3{color[0] / 255.0f, color[1] / 255.0f,
                        color[2] / 255.0f};
    }

    set.positions.emplace_back(x, y, z);
    set.normals.emplace_back(nx, ny, nz);
    set.sh_dc.emplace_back(fdc);
    set.sh_rest.push_back(std::move(fRest));
    set.opacities.push_back(opacity);
    set.scales.emplace_back(scale);
    set.rotations.emplace_back(Magnum::Vector3(rot[1], rot[2], rot[3]),
                               rot[0]);
    set.rotationsR.emplace_back(Mn::Vector3{0.0f}, 1.0f);
    set.times.push_back(time);
    set.timeScales.push_back(timeScale);
    set.motions.emplace_back(motion);
    std::array<float, 9> coeffs{};
    coeffs[0] = motion[0];
    coeffs[1] = motion[1];
    coeffs[2] = motion[2];
    set.motionCoeffs.push_back(coeffs);
    if (hasTemporal) {
      is4D_ = true;
    }
  }

  if (is4D_) {
    foregroundData_ = std::move(set);
    backgroundData_ = GaussianSetData{};
  } else {
    backgroundData_ = std::move(set);
    foregroundData_ = GaussianSetData{};
  }
  isHybrid_ = false;
  rebuildCombinedBuffers();

  ESP_DEBUG() << "Successfully read" << gaussianCount_ << "Gaussians";
  return true;
}

}  // namespace assets
}  // namespace esp
