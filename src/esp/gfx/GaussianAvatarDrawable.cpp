#include "GaussianAvatarDrawable.h"

#include <Corrade/Utility/Debug.h>

#include "GaussianSplattingRenderer.h"
#include "RenderTarget.h"
#include "esp/core/Check.h"

namespace esp {
namespace gfx {

GaussianAvatarDrawable::GaussianAvatarDrawable(
    scene::SceneNode& node,
    std::shared_ptr<assets::GaussianAvatarData> avatarData,
    DrawableConfiguration& cfg)
    : Drawable{node, nullptr, DrawableType::None, cfg,
               Magnum::Resource<LightSetup>()},
      avatarData_{std::move(avatarData)} {
  ESP_CHECK(avatarData_,
            "GaussianAvatarDrawable::GaussianAvatarDrawable(): data is null");

  renderer_ = std::make_unique<GaussianSplattingRenderer>();
  renderer_->initializeAvatar(*avatarData_);

  ESP_DEBUG() << "GaussianAvatarDrawable created with"
              << avatarData_->getPointCount() << "points";
}

GaussianAvatarDrawable::~GaussianAvatarDrawable() = default;

void GaussianAvatarDrawable::renderGaussians(
    RenderTarget& renderTarget,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& resolution,
    const Magnum::Vector3& cameraPos,
    float nearPlane,
    float farPlane,
    float time) {
  ESP_CHECK(renderer_ && renderer_->isInitialized(),
            "GaussianAvatarDrawable::renderGaussians(): renderer not initialized");

  auto& depthTexture = renderTarget.getCudaDepthTexture();
  bool hasColor = renderTarget.hasColorAttachment();

  if (hasColor) {
    auto& colorTexture = renderTarget.getGaussianColorTexture();
    renderer_->render(&colorTexture, &depthTexture, viewMatrix, projMatrix,
                      resolution, cameraPos, nearPlane, farPlane, time, true,
                      true);
  } else {
    renderer_->render(nullptr, &depthTexture, viewMatrix, projMatrix,
                      resolution, cameraPos, nearPlane, farPlane, time, false,
                      true);
  }
}

void GaussianAvatarDrawable::draw(const Magnum::Matrix4& transformationMatrix,
                                  Magnum::SceneGraph::Camera3D& camera) {
#ifdef ESP_BUILD_WITH_CUDA
  static_cast<void>(transformationMatrix);
  static_cast<void>(camera);
#else
  static_cast<void>(camera);
  static_cast<void>(transformationMatrix);
#endif
}

}  // namespace gfx
}  // namespace esp
