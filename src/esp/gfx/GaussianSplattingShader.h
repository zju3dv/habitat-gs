#pragma once

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Texture.h>

namespace esp {
namespace gfx {

/**
 * @brief Simple shader to write Gaussian Splatting depth (stored in an R32F
 * texture) into the primary depth buffer.
 */
class GaussianSplattingShader : public Magnum::GL::AbstractShaderProgram {
 public:
  static constexpr int DepthTextureUnit = 0;
  static constexpr int ColorTextureUnit = 1;

  explicit GaussianSplattingShader();

  GaussianSplattingShader& bindDepthTexture(Magnum::GL::Texture2D& texture);
  GaussianSplattingShader& bindColorTexture(Magnum::GL::Texture2D& texture);
  GaussianSplattingShader& setHasColor(bool hasColor);

 private:
  int depthTextureUniform_ = 0;
  int colorTextureUniform_ = 0;
  int hasColorUniform_ = 0;
};

}  // namespace gfx
}  // namespace esp
