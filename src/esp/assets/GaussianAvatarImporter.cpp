#include "GaussianAvatarImporter.h"

#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/Path.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "GaussianAvatarData.h"
#include "esp/core/Esp.h"
// Use the miniz implementation bundled with assimp for NPZ zip handling.
#define MINIZ_HEADER_FILE_ONLY
#include "deps/assimp/contrib/zip/src/miniz.h"

namespace Cr = Corrade;

namespace esp {
namespace assets {

namespace {
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
  out.wordSize =
      static_cast<size_t>(std::strtoul(descr.c_str() + 2, nullptr, 10));

  const auto fortranPos = header.find("fortran_order");
  if (fortranPos != std::string::npos) {
    const auto fortranStr =
        header.substr(fortranPos, header.size() - fortranPos);
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

bool parseNpyBuffer(const unsigned char* ptr,
                    size_t size,
                    NpyArrayData& out) {
  if (size < 10 || std::memcmp(ptr, "\x93NUMPY", 6) != 0) {
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

  if (headerOffset + headerLen > size) {
    return false;
  }

  const std::string header(
      reinterpret_cast<const char*>(ptr + headerOffset), headerLen);
  if (!parseNpyHeader(header, out)) {
    return false;
  }

  const size_t dataOffset = headerOffset + headerLen;
  size_t expectedSize = out.wordSize;
  for (size_t dim : out.shape) {
    expectedSize *= dim;
  }

  if (dataOffset + expectedSize > size) {
    return false;
  }

  out.data.assign(ptr + dataOffset, ptr + dataOffset + expectedSize);
  return true;
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

  bool ok = parseNpyBuffer(static_cast<unsigned char*>(buffer),
                           uncompressedSize, out);
  mz_free(buffer);
  return ok;
}

bool readFloatArray(const NpyArrayData& arr,
                    const std::string& name,
                    std::vector<float>& out) {
  if (arr.typeChar != 'f' ||
      (arr.wordSize != sizeof(float) && arr.wordSize != sizeof(double))) {
    ESP_ERROR() << "GaussianAvatarImporter:" << name
                << "has unsupported dtype or size (type=" << arr.typeChar
                << ", bytes=" << arr.wordSize << ").";
    return false;
  }

  out.resize(arr.elementCount());
  if (out.empty()) {
    return true;
  }

  const size_t dims = arr.shape.size();
  const unsigned char* raw = arr.data.data();

  if (!arr.fortran || dims <= 1) {
    if (arr.wordSize == sizeof(float)) {
      std::memcpy(out.data(), raw, out.size() * sizeof(float));
    } else {
      const double* src = reinterpret_cast<const double*>(raw);
      for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(src[i]);
      }
    }
    return true;
  }

  std::vector<size_t> strideF(dims, 1);
  for (size_t i = 1; i < dims; ++i) {
    strideF[i] = strideF[i - 1] * arr.shape[i - 1];
  }
  std::vector<size_t> strideC(dims, 1);
  for (size_t i = dims - 1; i > 0; --i) {
    strideC[i - 1] = strideC[i] * arr.shape[i];
  }

  const size_t elementCount = out.size();
  auto convertFortran = [&](const auto* src) {
    for (size_t cIdx = 0; cIdx < elementCount; ++cIdx) {
      size_t tmp = cIdx;
      size_t fIdx = 0;
      for (size_t d = 0; d < dims; ++d) {
        const size_t coord = tmp / strideC[d];
        tmp -= coord * strideC[d];
        fIdx += coord * strideF[d];
      }
      out[cIdx] = static_cast<float>(src[fIdx]);
    }
  };

  if (arr.wordSize == sizeof(float)) {
    convertFortran(reinterpret_cast<const float*>(raw));
  } else {
    convertFortran(reinterpret_cast<const double*>(raw));
  }
  return true;
}

int inferShDegree(size_t coeffCount) {
  const double root = std::sqrt(static_cast<double>(coeffCount));
  const int rootInt = static_cast<int>(std::round(root));
  if (rootInt * rootInt != static_cast<int>(coeffCount)) {
    return -1;
  }
  return rootInt - 1;
}
}  // namespace

std::shared_ptr<GaussianAvatarData> GaussianAvatarImporter::load(
    const LoadOptions& options) {
  if (options.canonicalGaussiansPath.empty()) {
    ESP_ERROR() << "GaussianAvatarImporter: missing canonical gaussians path.";
    return nullptr;
  }

  if (!Cr::Utility::Path::exists(options.canonicalGaussiansPath.c_str())) {
    ESP_ERROR() << "GaussianAvatarImporter: input file not found."
                << options.canonicalGaussiansPath;
    return nullptr;
  }

  mz_zip_archive zip{};
  if (!mz_zip_reader_init_file(&zip, options.canonicalGaussiansPath.c_str(), 0)) {
    ESP_ERROR() << "GaussianAvatarImporter: failed to open NPZ:"
                << options.canonicalGaussiansPath;
    return nullptr;
  }

  NpyArrayData meansNpy;
  NpyArrayData shsNpy;
  NpyArrayData scalesNpy;
  NpyArrayData quatsNpy;
  NpyArrayData opacitiesNpy;
  NpyArrayData lbsNpy;
  NpyArrayData invBindNpy;

  if (!loadNpyFromZip(zip, "means", meansNpy) ||
      !loadNpyFromZip(zip, "shs", shsNpy) ||
      !loadNpyFromZip(zip, "scales", scalesNpy) ||
      !loadNpyFromZip(zip, "quats", quatsNpy) ||
      !loadNpyFromZip(zip, "opacities", opacitiesNpy) ||
      !loadNpyFromZip(zip, "lbs_weights", lbsNpy) ||
      !loadNpyFromZip(zip, "joints_inv_bind_matrix", invBindNpy)) {
    ESP_ERROR() << "GaussianAvatarImporter: missing required entry in NPZ:"
                << options.canonicalGaussiansPath;
    mz_zip_reader_end(&zip);
    return nullptr;
  }
  mz_zip_reader_end(&zip);

  if (meansNpy.shape.size() != 2 || meansNpy.shape[1] != 3 ||
      meansNpy.typeChar != 'f' || meansNpy.wordSize != sizeof(float)) {
    ESP_ERROR() << "GaussianAvatarImporter: means has unexpected layout.";
    return nullptr;
  }

  const size_t pointCount = meansNpy.shape[0];

  size_t shCoeffCount = 0;
  if (shsNpy.shape.size() == 2 && shsNpy.shape[0] == pointCount &&
      shsNpy.shape[1] == 3) {
    shCoeffCount = 1;
  } else if (shsNpy.shape.size() == 3 && shsNpy.shape[0] == pointCount &&
             shsNpy.shape[2] == 3) {
    shCoeffCount = shsNpy.shape[1];
  } else {
    ESP_ERROR() << "GaussianAvatarImporter: shs has unexpected layout.";
    return nullptr;
  }

  const int shDegree = inferShDegree(shCoeffCount);
  if (shDegree < 0) {
    ESP_ERROR() << "GaussianAvatarImporter: shs coefficient count invalid.";
    return nullptr;
  }

  if (scalesNpy.shape.size() != 2 || scalesNpy.shape[0] != pointCount ||
      scalesNpy.shape[1] != 3) {
    ESP_ERROR() << "GaussianAvatarImporter: scales has unexpected layout.";
    return nullptr;
  }

  if (quatsNpy.shape.size() != 2 || quatsNpy.shape[0] != pointCount ||
      quatsNpy.shape[1] != 4) {
    ESP_ERROR() << "GaussianAvatarImporter: quats has unexpected layout.";
    return nullptr;
  }

  if ((opacitiesNpy.shape.size() != 1 &&
       opacitiesNpy.shape.size() != 2) ||
      opacitiesNpy.shape[0] != pointCount ||
      (opacitiesNpy.shape.size() == 2 && opacitiesNpy.shape[1] != 1)) {
    ESP_ERROR() << "GaussianAvatarImporter: opacities has unexpected layout.";
    return nullptr;
  }

  if (lbsNpy.shape.size() != 2 || lbsNpy.shape[0] != pointCount) {
    ESP_ERROR() << "GaussianAvatarImporter: lbs_weights has unexpected layout.";
    return nullptr;
  }
  const size_t jointCount = lbsNpy.shape[1];

  if (invBindNpy.shape.size() != 4 && invBindNpy.shape.size() != 3 &&
      invBindNpy.shape.size() != 2) {
    ESP_ERROR()
        << "GaussianAvatarImporter: joints_inv_bind_matrix has unexpected layout.";
    return nullptr;
  }
  if (invBindNpy.shape.size() == 4) {
    if (invBindNpy.shape[0] != 1 || invBindNpy.shape[1] != jointCount ||
        invBindNpy.shape[2] != 4 || invBindNpy.shape[3] != 4) {
      ESP_ERROR() << "GaussianAvatarImporter: inv bind size mismatch.";
      return nullptr;
    }
  } else if (invBindNpy.shape.size() == 3) {
    if (invBindNpy.shape[0] != jointCount || invBindNpy.shape[1] != 4 ||
        invBindNpy.shape[2] != 4) {
      ESP_ERROR() << "GaussianAvatarImporter: inv bind size mismatch.";
      return nullptr;
    }
  } else {
    if (invBindNpy.shape[0] != jointCount || invBindNpy.shape[1] != 16) {
      ESP_ERROR() << "GaussianAvatarImporter: inv bind size mismatch.";
      return nullptr;
    }
  }

  std::vector<float> means;
  std::vector<float> shs;
  std::vector<float> scales;
  std::vector<float> rotations;
  std::vector<float> opacities;
  std::vector<float> weights;
  std::vector<float> invBind;

  if (!readFloatArray(meansNpy, "means", means) ||
      !readFloatArray(shsNpy, "shs", shs) ||
      !readFloatArray(scalesNpy, "scales", scales) ||
      !readFloatArray(quatsNpy, "quats", rotations) ||
      !readFloatArray(opacitiesNpy, "opacities", opacities) ||
      !readFloatArray(lbsNpy, "lbs_weights", weights) ||
      !readFloatArray(invBindNpy, "joints_inv_bind_matrix", invBind)) {
    return nullptr;
  }

  if (shCoeffCount == 1 && !shs.empty()) {
    float minVal = shs[0];
    float maxVal = shs[0];
    for (float v : shs) {
      minVal = std::min(minVal, v);
      maxVal = std::max(maxVal, v);
    }
    if (minVal >= -1.0e-4f && maxVal <= 1.0001f) {
      constexpr float invShC0 = 1.0f / 0.28209479177387814f;
      for (float& v : shs) {
        v = (v - 0.5f) * invShC0;
      }
    }
  }

  auto data = std::make_shared<GaussianAvatarData>();
  data->setMeans(std::move(means));
  data->setShs(std::move(shs));
  data->setScales(std::move(scales));
  data->setRotations(std::move(rotations));
  data->setOpacities(std::move(opacities));
  data->setSkinningWeights(std::move(weights));
  data->setInvBindMatrices(std::move(invBind));
  data->setPointCount(static_cast<int>(pointCount));
  data->setJointCount(static_cast<int>(jointCount));
  data->setShDegree(shDegree);
  data->setShDim(static_cast<int>(shCoeffCount * 3));
  data->setSmplType(jointCount == 55 ? "smplx" : "smpl");

  ESP_DEBUG() << "GaussianAvatarImporter loaded" << pointCount << "points with"
              << jointCount << "joints. SH degree:" << shDegree;

  return data;
}

}  // namespace assets
}  // namespace esp
