#ifndef ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_
#define ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_

#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Matrix4.h>
#include <memory>

#include "Drawable.h"
#include "esp/assets/GaussianSplattingData.h"
#include "esp/core/Esp.h"

namespace esp {
namespace gfx {

class GaussianSplattingRenderer;

/**
 * @brief Drawable for rendering 3D Gaussian Splatting scenes.
 *
 * This drawable uses CUDA-based rasterization to render Gaussian splats
 * directly into OpenGL textures via CUDA-GL interop.
 */
class GaussianSplattingDrawable : public Drawable {
 public:
  /**
   * @brief Constructor
   *
   * @param node Scene node this drawable is attached to
   * @param gaussianData The Gaussian splatting data to render
   * @param normalizeQuaternion Whether to normalize quaternions on preprocess
   * @param group Drawable group this will be added to
   */
  GaussianSplattingDrawable(scene::SceneNode& node,
                            std::shared_ptr<assets::GaussianSplattingData> gaussianData,
                            DrawableConfiguration& cfg,
                            bool normalizeQuaternion = true);

  /**
   * @brief Destructor
   */
  ~GaussianSplattingDrawable() override;

  /**
   * @brief Render the Gaussian splats using CUDA
   *
   * This method performs CUDA-based rasterization directly into the
   * RenderTarget's OpenGL textures.
   *
   * @param renderTarget The target to render into
   * @param camera The camera to render from
   */
  void renderGaussians(RenderTarget& renderTarget,
                      const Magnum::Matrix4& viewMatrix,
                      const Magnum::Matrix4& projMatrix,
                      const Magnum::Vector2i& resolution,
                      const Magnum::Vector3& cameraPos,
                      float nearPlane,
                      float farPlane,
                      float time);

  /**
   * @brief Get the Gaussian splatting data
   */
  std::shared_ptr<assets::GaussianSplattingData> getGaussianData() const {
    return gaussianData_;
  }

  GaussianSplattingRenderer* getRenderer() const { return renderer_.get(); }

  /**
   * @brief Check if this drawable has valid Gaussian data
   */
  bool hasValidData() const {
    return gaussianData_ && gaussianData_->getGaussianCount() > 0;
  }

 protected:
  /**
   * @brief Draw function called by scene graph.
   */
  void draw(const Magnum::Matrix4& transformationMatrix,
            Magnum::SceneGraph::Camera3D& camera) override;

 private:
  //! The Gaussian splatting data to render
  std::shared_ptr<assets::GaussianSplattingData> gaussianData_;

  //! CUDA-based renderer for Gaussian splats
  std::unique_ptr<GaussianSplattingRenderer> renderer_;
};

}  // namespace gfx
}  // namespace esp

#endif  // ESP_GFX_GAUSSIANSPLATTINGDRAWABLE_H_
