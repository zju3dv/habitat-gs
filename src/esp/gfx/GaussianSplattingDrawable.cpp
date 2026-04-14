#include "GaussianSplattingDrawable.h"

#include <Corrade/Utility/Debug.h>

#include "GaussianSplattingRenderer.h"
#include "RenderTarget.h"
#include "RenderCamera.h"
#include "GaussianSplattingShader.h"
#include "esp/core/Check.h"

namespace Cr = Corrade;

namespace esp {
namespace gfx {

GaussianSplattingDrawable::GaussianSplattingDrawable(
    scene::SceneNode& node,
    std::shared_ptr<assets::GaussianSplattingData> gaussianData,
    DrawableConfiguration& cfg,
    bool normalizeQuaternion)
    : Drawable{node, 
               nullptr,  // No GL mesh for Gaussian splatting
               DrawableType::None,
               cfg,
               Magnum::Resource<LightSetup>()},
      gaussianData_{std::move(gaussianData)} {
  
  ESP_CHECK(gaussianData_,
            "GaussianSplattingDrawable::GaussianSplattingDrawable(): "
            "Gaussian data is null");
  
  // Create and initialize the CUDA renderer
  renderer_ = std::make_unique<GaussianSplattingRenderer>();
  renderer_->initialize(*gaussianData_, normalizeQuaternion);
  
  ESP_DEBUG() << "GaussianSplattingDrawable created with"
              << gaussianData_->getGaussianCount() << "Gaussians";
}

GaussianSplattingDrawable::~GaussianSplattingDrawable() = default;

void GaussianSplattingDrawable::renderGaussians(
    RenderTarget& renderTarget,
    const Magnum::Matrix4& viewMatrix,
    const Magnum::Matrix4& projMatrix,
    const Magnum::Vector2i& resolution,
    const Magnum::Vector3& cameraPos,
    float nearPlane,
    float farPlane,
    float time) {
  
  ESP_CHECK(renderer_ && renderer_->isInitialized(),
            "GaussianSplattingDrawable::renderGaussians(): "
            "Renderer not initialized");
  
  // Use CUDA-compatible depth texture (R32F format) instead of regular depth texture
  // Regular depth texture uses DepthComponent32F which cannot be registered with CUDA
  auto& depthTexture = renderTarget.getCudaDepthTexture();
  
  // Check if render target has color attachment
  bool hasColor = renderTarget.hasColorAttachment();
  
  if (hasColor) {
    // RGB sensor: render both color and depth to dedicated Gaussian targets
    auto& colorTexture = renderTarget.getGaussianColorTexture();
    renderer_->render(
        &colorTexture,   // Render color
        &depthTexture,   // Render depth
        viewMatrix,
        projMatrix,
        resolution,
        cameraPos,
        nearPlane,
        farPlane,
        time,
        true,   // renderColor = true
        true);  // renderDepth = true
  } else {
    // Depth-only sensor: render only depth (skip color computation)
    renderer_->render(
        nullptr,         // No color texture
        &depthTexture,   // Render depth only
        viewMatrix,
        projMatrix,
        resolution,
        cameraPos,
        nearPlane,
        farPlane,
        time,
        false,  // renderColor = false
        true);  // renderDepth = true
  }
}

void GaussianSplattingDrawable::draw(
    const Magnum::Matrix4& transformationMatrix,
    Magnum::SceneGraph::Camera3D& camera) {
#ifdef ESP_BUILD_WITH_CUDA
  static_cast<void>(transformationMatrix);
  auto* renderCamera = dynamic_cast<RenderCamera*>(&camera);
  if (!renderCamera) {
    return;
  }

  auto* renderTarget = renderCamera->getRenderTarget();
  auto* shader = renderCamera->getGaussianSplattingShader();
  if (!renderTarget || !shader) {
    return;
  }

  const Magnum::Matrix4 cameraWorld = renderCamera->object().absoluteTransformationMatrix();
  Magnum::Matrix4 projMatrix = renderCamera->projectionMatrix();
  Magnum::Vector2i resolution = renderCamera->viewport();
  Magnum::Vector3 cameraPos = cameraWorld.translation();
  auto depthRange = renderCamera->getDepthRange();

  float nearPlane = depthRange.first;
  float farPlane = depthRange.second;
  if (farPlane <= nearPlane) {
    // Fall back to extracting from projection matrix if not provided
    nearPlane = projMatrix.perspectiveProjectionNear();
    farPlane = projMatrix.perspectiveProjectionFar();
  }

  if (farPlane <= nearPlane) {
    return;
  }

  renderGaussians(*renderTarget, cameraWorld, projMatrix, resolution,
                  cameraPos, nearPlane, farPlane,
                  renderCamera->getGaussianTime());
  renderTarget->compositeGaussian(*shader, renderTarget->hasColorAttachment());
#else
  static_cast<void>(camera);
  static_cast<void>(transformationMatrix);
#endif
}

}  // namespace gfx
}  // namespace esp
