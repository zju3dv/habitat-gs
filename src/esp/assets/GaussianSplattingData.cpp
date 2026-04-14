#include "GaussianSplattingData.h"

#include <Corrade/Utility/Debug.h>
#include <algorithm>

namespace esp {
namespace assets {

void GaussianSplattingData::uploadBuffersToGPU(bool forceReload) {
  static bool warned = false;
  if (!warned) {
    ESP_WARNING()
        << "GaussianSplattingData::uploadBuffersToGPU is disabled to avoid "
           "duplicating 3DGS data in OpenGL buffers. Renderer consumes CUDA "
           "buffers directly.";
    warned = true;
  }
  // Intentionally no-op: 3DGS rendering uses CUDA allocations only.
  static_cast<void>(forceReload);
}

void GaussianSplattingData::addGaussian(GaussianSplat&& splat) {
  addStaticGaussian(std::move(splat));
}

void GaussianSplattingData::addStaticGaussian(GaussianSplat&& splat) {
  hasRotationR_ = hasRotationR_ || splat.hasRotationR;
  maxShRestCount_ = std::max(maxShRestCount_, splat.f_rest.size());
  shRestCountStatic_ = std::max(shRestCountStatic_, splat.f_rest.size());
  gaussians3D_.push_back(std::move(splat));
  if (layout_ == Layout::k4D && !gaussians4D_.empty()) {
    layout_ = Layout::kHybrid;
  }
  combinedDirty_ = true;
}

void GaussianSplattingData::addDynamicGaussian(GaussianSplat4D&& splat) {
  motionStride_ = std::max(motionStride_, splat.motionDim);
  hasRotationR_ = hasRotationR_ || splat.hasRotationR;
  maxShRestCount_ = std::max(maxShRestCount_, splat.f_rest.size());
  shRestCountDynamic_ = std::max(shRestCountDynamic_, splat.f_rest.size());
  gaussians4D_.push_back(std::move(splat));
  if (layout_ == Layout::k3D && !gaussians3D_.empty()) {
    layout_ = Layout::kHybrid;
  } else {
    layout_ = Layout::k4D;
  }
  combinedDirty_ = true;
}

const std::vector<GaussianSplat>& GaussianSplattingData::getGaussians() const {
  if (!combinedDirty_) {
    return combinedGaussians_;
  }

  combinedGaussians_.clear();
  combinedGaussians_.reserve(gaussians3D_.size() + gaussians4D_.size());

  auto copyBaseSplat = [](const GaussianSplat& src) {
    GaussianSplat dst;
    dst.position = src.position;
    dst.normal = src.normal;
    dst.f_dc = src.f_dc;
    dst.opacity = src.opacity;
    dst.scale = src.scale;
    dst.rotation = src.rotation;
    dst.rotationR = src.rotationR;
    dst.hasRotationR = src.hasRotationR;
    dst.f_rest = Corrade::Containers::Array<float>(src.f_rest.size());
    for (size_t i = 0; i < src.f_rest.size(); ++i) {
      dst.f_rest[i] = src.f_rest[i];
    }
    return dst;
  };

  for (const auto& g3d : gaussians3D_) {
    combinedGaussians_.push_back(copyBaseSplat(g3d));
  }

  for (const auto& g4d : gaussians4D_) {
    combinedGaussians_.push_back(copyBaseSplat(g4d));
  }
  combinedDirty_ = false;
  return combinedGaussians_;
}

}  // namespace assets
}  // namespace esp
