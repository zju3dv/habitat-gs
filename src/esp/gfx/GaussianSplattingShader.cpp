#include "GaussianSplattingShader.h"

#include <Corrade/Utility/Assert.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>

namespace Mn = Magnum;

namespace esp {
namespace gfx {

GaussianSplattingShader::GaussianSplattingShader() {
  Mn::GL::Shader vert{Mn::GL::Version::GL410,
                      Mn::GL::Shader::Type::Vertex};
  Mn::GL::Shader frag{Mn::GL::Version::GL410,
                      Mn::GL::Shader::Type::Fragment};

  vert.addSource(
      "const vec2 positions[3] = vec2[3](\n"
      "    vec2(-1.0, -1.0),\n"
      "    vec2(3.0, -1.0),\n"
      "    vec2(-1.0, 3.0));\n"
      "out vec2 vTexCoord;\n"
      "void main() {\n"
      "  vTexCoord = positions[gl_VertexID] * 0.5 + 0.5;\n"
      "  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);\n"
      "}\n");

  frag.addSource(
      "in vec2 vTexCoord;\n"
      "uniform sampler2D depthTexture;\n"
      "uniform sampler2D colorTexture;\n"
      "uniform bool hasColor;\n"
      "layout(location = 0) out vec4 fragColor;\n"
      "void main() {\n"
      "  float depth = texture(depthTexture, vTexCoord).r;\n"
      "  if(depth >= 1.0) {\n"
      "    discard;\n"
      "  }\n"
      "  gl_FragDepth = depth;\n"
      "  if(hasColor) {\n"
      "    fragColor = texture(colorTexture, vTexCoord);\n"
      "  } else {\n"
      "    fragColor = vec4(0.0);\n"
      "  }\n"
      "}\n");

  CORRADE_INTERNAL_ASSERT_OUTPUT(vert.compile());
  CORRADE_INTERNAL_ASSERT_OUTPUT(frag.compile());
  attachShaders({vert, frag});
  CORRADE_INTERNAL_ASSERT_OUTPUT(link());

  depthTextureUniform_ = uniformLocation("depthTexture");
  colorTextureUniform_ = uniformLocation("colorTexture");
  hasColorUniform_ = uniformLocation("hasColor");
  setUniform(depthTextureUniform_, DepthTextureUnit);
  setUniform(colorTextureUniform_, ColorTextureUnit);
}

GaussianSplattingShader& GaussianSplattingShader::bindDepthTexture(
    Mn::GL::Texture2D& texture) {
  texture.bind(DepthTextureUnit);
  return *this;
}

GaussianSplattingShader& GaussianSplattingShader::bindColorTexture(
    Mn::GL::Texture2D& texture) {
  texture.bind(ColorTextureUnit);
  return *this;
}

GaussianSplattingShader& GaussianSplattingShader::setHasColor(bool hasColor) {
  setUniform(hasColorUniform_, hasColor);
  return *this;
}

}  // namespace gfx
}  // namespace esp
