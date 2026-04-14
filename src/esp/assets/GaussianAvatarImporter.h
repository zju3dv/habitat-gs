#ifndef ESP_ASSETS_GAUSSIANAVATARIMPORTER_H_
#define ESP_ASSETS_GAUSSIANAVATARIMPORTER_H_

#include <memory>
#include <string>

#include "esp/core/Esp.h"

namespace esp {
namespace assets {

class GaussianAvatarData;

/**
 * @brief Loader for Gaussian Avatar canonical Gaussian assets.
 */
class GaussianAvatarImporter {
 public:
  struct LoadOptions {
    std::string canonicalGaussiansPath;
  };

  GaussianAvatarImporter() = default;

  /**
   * @brief Load canonical avatar data from disk.
   * @return Shared pointer to GaussianAvatarData on success, nullptr otherwise.
   */
  std::shared_ptr<GaussianAvatarData> load(const LoadOptions& options);
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_GAUSSIANAVATARIMPORTER_H_
