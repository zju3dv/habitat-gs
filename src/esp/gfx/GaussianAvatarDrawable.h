#ifndef ESP_GFX_GAUSSIANAVATARDRAWABLE_H_
#define ESP_GFX_GAUSSIANAVATARDRAWABLE_H_

#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Matrix4.h>
#include <memory>

#include "Drawable.h"
#include "esp/assets/GaussianAvatarData.h"
#include "esp/core/Esp.h"

namespace esp {
namespace gfx {

class GaussianSplattingRenderer;

/**
 * @brief Drawable for rendering Gaussian Avatar instances.
 */
class GaussianAvatarDrawable : public Drawable {
 public:
  GaussianAvatarDrawable(scene::SceneNode& node,
                         std::shared_ptr<assets::GaussianAvatarData> avatarData,
                         DrawableConfiguration& cfg);

  ~GaussianAvatarDrawable() override;

  void renderGaussians(RenderTarget& renderTarget,
                       const Magnum::Matrix4& viewMatrix,
                       const Magnum::Matrix4& projMatrix,
                       const Magnum::Vector2i& resolution,
                       const Magnum::Vector3& cameraPos,
                       float nearPlane,
                       float farPlane,
                       float time);

  std::shared_ptr<assets::GaussianAvatarData> getAvatarData() const {
    return avatarData_;
  }

  GaussianSplattingRenderer* getRenderer() const { return renderer_.get(); }

 protected:
  void draw(const Magnum::Matrix4& transformationMatrix,
            Magnum::SceneGraph::Camera3D& camera) override;

 private:
  std::shared_ptr<assets::GaussianAvatarData> avatarData_;
  std::unique_ptr<GaussianSplattingRenderer> renderer_;
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANAVATARDRAWABLE_H_
