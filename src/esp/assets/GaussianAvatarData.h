#ifndef ESP_ASSETS_GAUSSIANAVATARDATA_H_
#define ESP_ASSETS_GAUSSIANAVATARDATA_H_

#include <cstddef>
#include <string>
#include <vector>

#include "esp/core/Esp.h"

namespace esp {
namespace assets {

/**
 * @brief Canonical data container for a Gaussian Avatar instance.
 *
 * Stores canonical Gaussian data and skinning weights on the CPU.
 * GPU uploads are handled by the renderer at initialization time.
 */
class GaussianAvatarData {
 public:
  GaussianAvatarData() = default;

  void setMeans(std::vector<float> means) { means_ = std::move(means); }
  void setShs(std::vector<float> shs) { shs_ = std::move(shs); }
  void setScales(std::vector<float> scales) { scales_ = std::move(scales); }
  void setRotations(std::vector<float> rotations) {
    rotations_ = std::move(rotations);
  }
  void setOpacities(std::vector<float> opacities) {
    opacities_ = std::move(opacities);
  }
  void setSkinningWeights(std::vector<float> weights) {
    skinningWeights_ = std::move(weights);
  }
  void setInvBindMatrices(std::vector<float> invBind) {
    invBindMatrices_ = std::move(invBind);
  }

  const std::vector<float>& getMeans() const { return means_; }
  const std::vector<float>& getShs() const { return shs_; }
  const std::vector<float>& getScales() const { return scales_; }
  const std::vector<float>& getRotations() const { return rotations_; }
  const std::vector<float>& getOpacities() const { return opacities_; }
  const std::vector<float>& getSkinningWeights() const {
    return skinningWeights_;
  }
  const std::vector<float>& getInvBindMatrices() const {
    return invBindMatrices_;
  }

  void applyScale(float scale) {
    if (scale == 1.0f) {
      return;
    }
    for (float& value : scales_) {
      value *= scale;
    }
  }

  void setPointCount(int count) { pointCount_ = count; }
  void setJointCount(int count) { jointCount_ = count; }
  void setShDegree(int degree) { shDegree_ = degree; }
  void setShDim(int dim) { shDim_ = dim; }
  void setSmplType(const std::string& type) { smplType_ = type; }

  int getPointCount() const { return pointCount_; }
  int getJointCount() const { return jointCount_; }
  int getShDegree() const { return shDegree_; }
  int getShDim() const { return shDim_; }
  const std::string& getSmplType() const { return smplType_; }

  bool hasData() const {
    return pointCount_ > 0 && jointCount_ > 0 &&
           means_.size() == static_cast<size_t>(pointCount_) * 3 &&
           shs_.size() == static_cast<size_t>(pointCount_) * shDim_ &&
           scales_.size() == static_cast<size_t>(pointCount_) * 3 &&
           rotations_.size() == static_cast<size_t>(pointCount_) * 4 &&
           opacities_.size() == static_cast<size_t>(pointCount_) &&
           skinningWeights_.size() ==
               static_cast<size_t>(pointCount_) * jointCount_ &&
           invBindMatrices_.size() ==
               static_cast<size_t>(jointCount_) * 16;
  }

 private:
  std::vector<float> means_;
  std::vector<float> shs_;
  std::vector<float> scales_;
  std::vector<float> rotations_;
  std::vector<float> opacities_;
  std::vector<float> skinningWeights_;
  std::vector<float> invBindMatrices_;
  int pointCount_ = 0;
  int jointCount_ = 0;
  int shDegree_ = 0;
  int shDim_ = 3;
  std::string smplType_{"smpl"};
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_GAUSSIANAVATARDATA_H_
