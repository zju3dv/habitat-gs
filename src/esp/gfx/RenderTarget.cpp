// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <Corrade/Containers/StridedArrayView.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/BufferImage.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Color.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Shaders/GenericGL.h>

#include "RenderTarget.h"
#include "GaussianSplattingShader.h"
#include "esp/sensor/VisualSensor.h"

#include "esp/gfx_batch/DepthUnprojection.h"
#include "esp/gfx_batch/Hbao.h"

#ifdef ESP_BUILD_WITH_CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "HelperCuda.h"
#endif

namespace Cr = Corrade;
namespace Mn = Magnum;

namespace esp {
namespace gfx {

const Mn::GL::Framebuffer::ColorAttachment RgbaBufferAttachment =
    Mn::GL::Framebuffer::ColorAttachment{0};
const Mn::GL::Framebuffer::ColorAttachment ObjectIdTextureColorAttachment =
    Mn::GL::Framebuffer::ColorAttachment{1};
const Mn::GL::Framebuffer::ColorAttachment UnprojectedDepthBufferAttachment =
    Mn::GL::Framebuffer::ColorAttachment{0};

struct RenderTarget::Impl {
  Impl(const Mn::Vector2i& size,
       const Mn::Vector2& depthUnprojection,
       gfx_batch::DepthShader* depthShader,
       Flags flags,
       const sensor::VisualSensor* visualSensor)
      : colorBuffer_{},
        colorTexture_{},
        objectIdTexture_{},
        depthRenderTexture_{},
        framebuffer_{Mn::NoCreate},
        useColorTexture_{false},
        depthUnprojection_{depthUnprojection},
        depthShader_{depthShader},
        unprojectedDepth_{Mn::NoCreate},
        depthUnprojectionMesh_{Mn::NoCreate},
        depthUnprojectionFrameBuffer_{Mn::NoCreate},
        flags_{flags},
        visualSensor_{visualSensor} {
    if (depthShader_) {
      CORRADE_INTERNAL_ASSERT(
          depthShader_->flags() &
          gfx_batch::DepthShader::Flag::UnprojectExistingDepth);
    }

#ifdef ESP_BUILD_WITH_CUDA
    // Use texture for RGBA when CUDA is available for better interop
    useColorTexture_ = true;
    
    // Create CUDA-compatible depth texture (R32F format)
    // This is required for Gaussian Splatting rendering as CUDA cannot register depth formats
    cudaDepthTexture_.setMinificationFilter(Mn::GL::SamplerFilter::Nearest)
        .setMagnificationFilter(Mn::GL::SamplerFilter::Nearest)
        .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
        .setStorage(1, Mn::GL::TextureFormat::R32F, size);
#endif

    if (flags_ & Flag::RgbaAttachment) {
      if (useColorTexture_) {
        // Use RGBA32F for CUDA compatibility (Gaussian Splatting outputs float)
        colorTexture_.setMinificationFilter(Mn::GL::SamplerFilter::Linear)
            .setMagnificationFilter(Mn::GL::SamplerFilter::Linear)
            .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
            .setStorage(1, Mn::GL::TextureFormat::RGBA32F, size);
#ifdef ESP_BUILD_WITH_CUDA
        gaussianColorTexture_
            .setMinificationFilter(Mn::GL::SamplerFilter::Linear)
            .setMagnificationFilter(Mn::GL::SamplerFilter::Linear)
            .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
            .setStorage(1, Mn::GL::TextureFormat::RGBA32F, size);
#endif
      } else {
        colorBuffer_.setStorage(Mn::GL::RenderbufferFormat::RGBA8, size);
      }
    }
    if (flags_ & Flag::ObjectIdAttachment) {
      objectIdTexture_.setMinificationFilter(Mn::GL::SamplerFilter::Nearest)
          .setMagnificationFilter(Mn::GL::SamplerFilter::Nearest)
          .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
          .setStorage(1, Mn::GL::TextureFormat::R32UI, size);
    }

    if (flags_ & Flag::DepthTextureAttachment) {
      depthRenderTexture_.setMinificationFilter(Mn::GL::SamplerFilter::Nearest)
          .setMagnificationFilter(Mn::GL::SamplerFilter::Nearest)
          .setWrapping(Mn::GL::SamplerWrapping::ClampToEdge)
          .setStorage(1, Mn::GL::TextureFormat::DepthComponent32F, size);
    } else {
      // we use the unprojectedDepth_ as the depth buffer
      unprojectedDepth_ = Mn::GL::Renderbuffer{};
      unprojectedDepth_.setStorage(Mn::GL::RenderbufferFormat::DepthComponent24,
                                   size);
    }

    framebuffer_ = Mn::GL::Framebuffer{{{}, size}};
    if (flags_ & Flag::RgbaAttachment) {
      if (useColorTexture_) {
        framebuffer_.attachTexture(RgbaBufferAttachment, colorTexture_, 0);
      } else {
        framebuffer_.attachRenderbuffer(RgbaBufferAttachment, colorBuffer_);
      }
    }
    if (flags_ & Flag::ObjectIdAttachment) {
      framebuffer_.attachTexture(ObjectIdTextureColorAttachment,
                                 objectIdTexture_, 0);
    }
    if (flags_ & Flag::DepthTextureAttachment) {
      framebuffer_.attachTexture(Mn::GL::Framebuffer::BufferAttachment::Depth,
                                 depthRenderTexture_, 0);
    } else {
      framebuffer_.attachRenderbuffer(
          Mn::GL::Framebuffer::BufferAttachment::Depth, unprojectedDepth_);
    }

    framebuffer_.mapForDraw(
        {{Mn::Shaders::GenericGL3D::ColorOutput,
          (flags_ & Flag::RgbaAttachment
               ? RgbaBufferAttachment
               : Mn::GL::Framebuffer::DrawAttachment::None)},
         {Mn::Shaders::GenericGL3D::ObjectIdOutput,
          (flags_ & Flag::ObjectIdAttachment
               ? ObjectIdTextureColorAttachment
               : Mn::GL::Framebuffer::DrawAttachment::None)}});

    CORRADE_INTERNAL_ASSERT(
        framebuffer_.checkStatus(Mn::GL::FramebufferTarget::Draw) ==
        Mn::GL::Framebuffer::Status::Complete);

    if (flags_ & Flag::HorizonBasedAmbientOcclusion) {
      // depth texture is required for HBAO
      CORRADE_INTERNAL_ASSERT(flags_ & Flag::DepthTextureAttachment);
      // TODO Drive construction based on premade configurations
      hbao_ = gfx_batch::Hbao{
          gfx_batch::HbaoConfiguration{}
              .setSize(size)
              .setUseSpecialBlur(true)
              .setUseLayeredGeometryShader(true)
          // TODO other options here?
      };
    }
  }

  void initDepthUnprojector() {
    CORRADE_ASSERT(
        flags_ & Flag::DepthTextureAttachment,
        "RenderTarget::Impl::initDepthUnprojector(): this render target "
        "was not created with depth texture enabled.", );

    if (depthUnprojectionMesh_.id() == 0) {
      unprojectedDepth_ = Mn::GL::Renderbuffer{};
      unprojectedDepth_.setStorage(Mn::GL::RenderbufferFormat::R32F,
                                   framebufferSize());

      depthUnprojectionFrameBuffer_ =
          Mn::GL::Framebuffer{{{}, framebufferSize()}};
      depthUnprojectionFrameBuffer_
          .attachRenderbuffer(UnprojectedDepthBufferAttachment,
                              unprojectedDepth_)
          .mapForDraw({{0, UnprojectedDepthBufferAttachment}});
      CORRADE_INTERNAL_ASSERT(
          framebuffer_.checkStatus(Mn::GL::FramebufferTarget::Draw) ==
          Mn::GL::Framebuffer::Status::Complete);

      depthUnprojectionMesh_ = Mn::GL::Mesh{};
      depthUnprojectionMesh_.setCount(3);
    }
  }

  void unprojectDepthGPU() {
    CORRADE_INTERNAL_ASSERT(depthShader_ != nullptr);
    CORRADE_ASSERT(
        flags_ & Flag::DepthTextureAttachment,
        "RenderTarget::Impl::unprojectDepthGPU(): this render target "
        "was not created with depth texture enabled.", );
    initDepthUnprojector();

    depthUnprojectionFrameBuffer_.bind();
    (*depthShader_)
        .bindDepthTexture(depthRenderTexture_)
        .setDepthUnprojection(depthUnprojection_)
        .draw(depthUnprojectionMesh_);
  }

  void renderEnter() {
    hadGaussianCompositeInLastPass_ = false;
    framebuffer_.clearDepth(1.0);
    if (flags_ & Flag::RgbaAttachment) {
      if (visualSensor_) {
        framebuffer_.clearColor(0, static_cast<esp::sensor::VisualSensorSpec*>(
                                       visualSensor_->specification().get())
                                       ->clearColor);
      } else {
        framebuffer_.clearColor(0, Mn::Color4{0, 0, 0, 1});
      }
    }
    if (flags_ & Flag::ObjectIdAttachment) {
      framebuffer_.clearColor(1, Mn::Vector4ui{});
    }
    framebuffer_.bind();
  }

  void renderReEnter() { framebuffer_.bind(); }

  void renderExit() {}

  void tryDrawHbao() {
    if (!hbao_) {
      return;
    }

    hbao_->drawEffect(visualSensor_->getProjectionMatrix(),
                      gfx_batch::HbaoType::CacheAware, depthRenderTexture_,
                      framebuffer_);
  }

  void blitRgbaTo(Mn::GL::AbstractFramebuffer& target,
                  const Mn::Range2Di& targetRectangle) {
    CORRADE_ASSERT(
        flags_ & Flag::RgbaAttachment,
        "RenderTarget::Impl::blitRgbaToDefault(): this render target "
        "was not created with rgba render buffer enabled.", );

    framebuffer_.mapForRead(RgbaBufferAttachment);
    Mn::GL::AbstractFramebuffer::blit(
        framebuffer_, target, framebuffer_.viewport(), targetRectangle,
        Mn::GL::FramebufferBlit::Color, Mn::GL::FramebufferBlitFilter::Linear);
  }

  bool blitRgbaTo(Impl& target) {
    if (!(flags_ & Flag::RgbaAttachment) ||
        !(target.flags_ & Flag::RgbaAttachment)) {
      return false;
    }
    if (framebufferSize() != target.framebufferSize()) {
      return false;
    }

    framebuffer_.mapForRead(RgbaBufferAttachment);
    Mn::GL::AbstractFramebuffer::blit(
        framebuffer_, target.framebuffer_, framebuffer_.viewport(),
        target.framebuffer_.viewport(), Mn::GL::FramebufferBlit::Color,
        Mn::GL::FramebufferBlitFilter::Linear);
    return true;
  }

  bool blitDepthTo(Impl& target) {
    if (!(flags_ & Flag::DepthTextureAttachment) ||
        !(target.flags_ & Flag::DepthTextureAttachment)) {
      return false;
    }
    if (framebufferSize() != target.framebufferSize()) {
      return false;
    }

    // Clear stale GL errors so we can reliably detect blit failures.
    while (glGetError() != GL_NO_ERROR) {
    }

    Mn::GL::AbstractFramebuffer::blit(
        framebuffer_, target.framebuffer_, framebuffer_.viewport(),
        target.framebuffer_.viewport(), Mn::GL::FramebufferBlit::Depth,
        Mn::GL::FramebufferBlitFilter::Nearest);

    return glGetError() == GL_NO_ERROR;
  }

  void readFrameRgba(const Mn::MutableImageView2D& view) {
    CORRADE_ASSERT(flags_ & Flag::RgbaAttachment,
                   "RenderTarget::Impl::readFrameRgba(): this render target "
                   "was not created with rgba render buffer enabled.", );

    if (useColorTexture_) {
      // Read from texture
      colorTexture_.image(0, view);
    } else {
      // Read from renderbuffer
      framebuffer_.mapForRead(RgbaBufferAttachment)
          .read(framebuffer_.viewport(), view);
    }
  }

  void readFrameDepth(const Mn::MutableImageView2D& view) {
    // Standard mesh rendering path: read from the depth render texture
    CORRADE_ASSERT(flags_ & Flag::DepthTextureAttachment,
                   "RenderTarget::Impl::readFrameDepth(): this render target "
                   "was not created with depth texture enabled.", );
    if (depthShader_) {
      unprojectDepthGPU();
      depthUnprojectionFrameBuffer_.mapForRead(UnprojectedDepthBufferAttachment)
          .read(framebuffer_.viewport(), view);
    } else {
      Mn::MutableImageView2D depthBufferView{
          Mn::GL::PixelFormat::DepthComponent, Mn::GL::PixelType::Float,
          view.size(), view.data()};
      framebuffer_.read(framebuffer_.viewport(), depthBufferView);
      gfx_batch::unprojectDepth(depthUnprojection_, view.pixels<Mn::Float>());
    }
  }

  void readFrameObjectId(const Mn::MutableImageView2D& view) {
    CORRADE_ASSERT(
        flags_ & Flag::ObjectIdAttachment,
        "RenderTarget::Impl::readFrameObjectId(): this render target "
        "was not created with objectId render texture enabled.", );
    framebuffer_.mapForRead(ObjectIdTextureColorAttachment)
        .read(framebuffer_.viewport(), view);
  }

  Mn::Vector2i framebufferSize() const {
    return framebuffer_.viewport().size();
  }

  Magnum::GL::Texture2D& getDepthTexture() {
    CORRADE_ASSERT(flags_ & Flag::DepthTextureAttachment,
                   "RenderTarget::Impl::getDepthTexture(): this render target "
                   "was not created with depth texture enabled.",
                   depthRenderTexture_);
    return depthRenderTexture_;
  }
  Magnum::GL::Texture2D& getObjectIdTexture() {
    CORRADE_ASSERT(
        flags_ & Flag::ObjectIdAttachment,
        "RenderTarget::Impl::getObjectIdTexture(): this render target "
        "was not created with object id texture enabled.",
        objectIdTexture_);
    return objectIdTexture_;
  }

  Magnum::GL::Texture2D& getColorTexture() {
    CORRADE_ASSERT(
        flags_ & Flag::RgbaAttachment && useColorTexture_,
        "RenderTarget::Impl::getColorTexture(): this render target "
        "was not created with color texture enabled.",
        colorTexture_);
    return colorTexture_;
  }

#ifdef ESP_BUILD_WITH_CUDA
  void compositeGaussian(GaussianSplattingShader& shader, bool hasColor) {
    if (gaussianDepthMesh_.id() == 0) {
      gaussianDepthMesh_ = Mn::GL::Mesh{};
      gaussianDepthMesh_.setCount(3);
    }

    framebuffer_.bind();
    bool toggleBlend = false;
    if (hasColor) {
      toggleBlend = !glIsEnabled(GL_BLEND);
      if (toggleBlend) {
        Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::Blending);
      }
      // Gaussian colors are premultiplied by alpha.
      Mn::GL::Renderer::setBlendFunction(
          Mn::GL::Renderer::BlendFunction::One,
          Mn::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
      Mn::GL::Renderer::setBlendEquation(
          Mn::GL::Renderer::BlendEquation::Add);
    }
    Mn::GL::Renderer::setDepthMask(true);
    Mn::GL::Renderer::setDepthFunction(
        Mn::GL::Renderer::DepthFunction::LessOrEqual);
    Mn::GL::Renderer::setColorMask(hasColor, hasColor, hasColor, hasColor);

    shader.setHasColor(hasColor).bindDepthTexture(cudaDepthTexture_);
    if (hasColor) {
      shader.bindColorTexture(gaussianColorTexture_);
    }
    shader.draw(gaussianDepthMesh_);

    Mn::GL::Renderer::setColorMask(true, true, true, true);
    Mn::GL::Renderer::setDepthFunction(
        Mn::GL::Renderer::DepthFunction::LessOrEqual);
    if (toggleBlend) {
      Mn::GL::Renderer::disable(Mn::GL::Renderer::Feature::Blending);
    }
    hadGaussianCompositeInLastPass_ = true;
  }
#endif
  
#ifdef ESP_BUILD_WITH_CUDA
  Magnum::GL::Texture2D& getCudaDepthTexture() {
    return cudaDepthTexture_;
  }

  Magnum::GL::Texture2D& getGaussianColorTexture() {
    CORRADE_ASSERT(
        flags_ & Flag::RgbaAttachment && useColorTexture_,
        "RenderTarget::Impl::getGaussianColorTexture(): this render target "
        "was not created with color texture enabled.",
        gaussianColorTexture_);
    return gaussianColorTexture_;
  }
#endif

  bool hasColorAttachment() const {
    return bool(flags_ & Flag::RgbaAttachment);
  }

  bool hadGaussianCompositeInLastPass() const {
    return hadGaussianCompositeInLastPass_;
  }

#ifdef ESP_BUILD_WITH_CUDA
  void readFrameRgbaGPU(uint8_t* devPtr) {
    // TODO: Consider implementing the GPU read functions with EGLImage
    // See discussion here:
    // https://github.com/facebookresearch/habitat-sim/pull/114#discussion_r312718502
    CORRADE_ASSERT(flags_ & Flag::RgbaAttachment,
                   "RenderTarget::Impl::readFrameRgbaGPU(): this render target "
                   "was not created with rgba render buffer enabled.", );

    if (colorBufferCugl_ == nullptr)
      checkCudaErrors(cudaGraphicsGLRegisterImage(
          &colorBufferCugl_, colorBuffer_.id(), GL_RENDERBUFFER,
          cudaGraphicsRegisterFlagsReadOnly));

    checkCudaErrors(cudaGraphicsMapResources(1, &colorBufferCugl_, 0));

    cudaArray* array = nullptr;
    checkCudaErrors(
        cudaGraphicsSubResourceGetMappedArray(&array, colorBufferCugl_, 0, 0));
    const int widthInBytes = framebufferSize().x() * 4 * sizeof(uint8_t);
    checkCudaErrors(cudaMemcpy2DFromArray(devPtr, widthInBytes, array, 0, 0,
                                          widthInBytes, framebufferSize().y(),
                                          cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &colorBufferCugl_, 0));
  }

  void readFrameDepthGPU(float* devPtr) {
    CORRADE_ASSERT(
        flags_ & Flag::DepthTextureAttachment,
        "RenderTarget::Impl::readFrameDepthGPU(): this render target "
        "was not created with depth texture enabled.", );
    unprojectDepthGPU();

    if (depthBufferCugl_ == nullptr)
      checkCudaErrors(cudaGraphicsGLRegisterImage(
          &depthBufferCugl_, unprojectedDepth_.id(), GL_RENDERBUFFER,
          cudaGraphicsRegisterFlagsReadOnly));

    checkCudaErrors(cudaGraphicsMapResources(1, &depthBufferCugl_, 0));

    cudaArray* array = nullptr;
    checkCudaErrors(
        cudaGraphicsSubResourceGetMappedArray(&array, depthBufferCugl_, 0, 0));
    const int widthInBytes = framebufferSize().x() * 1 * sizeof(float);
    checkCudaErrors(cudaMemcpy2DFromArray(devPtr, widthInBytes, array, 0, 0,
                                          widthInBytes, framebufferSize().y(),
                                          cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &depthBufferCugl_, 0));
  }

  void readFrameObjectIdGPU(int32_t* devPtr) {
    CORRADE_ASSERT(
        flags_ & Flag::ObjectIdAttachment,
        "RenderTarget::Impl::readFrameObjectIdGPU(): this render target "
        "was not created with objectId render texture enabled.", );

    // We replaced the render buffer by the render texture in #1203, and
    // expected NO performance loss. Let us know if it is not the case.
    if (objecIdBufferCugl_ == nullptr)
      checkCudaErrors(cudaGraphicsGLRegisterImage(
          &objecIdBufferCugl_, objectIdTexture_.id(), GL_TEXTURE_2D,
          cudaGraphicsRegisterFlagsReadOnly));

    checkCudaErrors(cudaGraphicsMapResources(1, &objecIdBufferCugl_, 0));

    cudaArray* array = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &array, objecIdBufferCugl_, 0, 0));
    const int widthInBytes = framebufferSize().x() * 1 * sizeof(int32_t);
    checkCudaErrors(cudaMemcpy2DFromArray(devPtr, widthInBytes, array, 0, 0,
                                          widthInBytes, framebufferSize().y(),
                                          cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &objecIdBufferCugl_, 0));
  }
#endif

#ifdef ESP_BUILD_WITH_CUDA
  ~Impl() {
    if (colorBufferCugl_ != nullptr)
      checkCudaErrors(cudaGraphicsUnregisterResource(colorBufferCugl_));
    if (depthBufferCugl_ != nullptr)
      checkCudaErrors(cudaGraphicsUnregisterResource(depthBufferCugl_));
    if (objecIdBufferCugl_ != nullptr)
      checkCudaErrors(cudaGraphicsUnregisterResource(objecIdBufferCugl_));
  }
#else
  ~Impl() = default;
#endif

 private:
  Mn::GL::Renderbuffer colorBuffer_;
  Mn::GL::Texture2D colorTexture_;  // Color texture for CUDA-GL interop
#ifdef ESP_BUILD_WITH_CUDA
  Mn::GL::Texture2D gaussianColorTexture_;  // Separate color for Gaussian pass
#endif
  Mn::GL::Texture2D objectIdTexture_;
  Mn::GL::Texture2D depthRenderTexture_;
  Mn::GL::Framebuffer framebuffer_;
  bool useColorTexture_ = false;  // Flag to use texture instead of renderbuffer
  bool hadGaussianCompositeInLastPass_ = false;
  
#ifdef ESP_BUILD_WITH_CUDA
  // CUDA-compatible depth texture (R32F format) for Gaussian Splatting
  // The regular depthRenderTexture_ uses DepthComponent32F which cannot be registered with CUDA
  Mn::GL::Texture2D cudaDepthTexture_;
#endif

  Mn::Vector2 depthUnprojection_;
  gfx_batch::DepthShader* depthShader_;
  Mn::GL::Renderbuffer unprojectedDepth_;
  Mn::GL::Mesh depthUnprojectionMesh_;
  Mn::GL::Framebuffer depthUnprojectionFrameBuffer_;
  Mn::GL::Mesh gaussianDepthMesh_{Mn::NoCreate};

  Flags flags_;

  const sensor::VisualSensor* visualSensor_ = nullptr;

#ifdef ESP_BUILD_WITH_CUDA
  cudaGraphicsResource_t colorBufferCugl_ = nullptr;
  cudaGraphicsResource_t objecIdBufferCugl_ = nullptr;
  cudaGraphicsResource_t depthBufferCugl_ = nullptr;
#endif

  Cr::Containers::Optional<gfx_batch::Hbao> hbao_{};
};  // namespace gfx

RenderTarget::RenderTarget(const Mn::Vector2i& size,
                           const Mn::Vector2& depthUnprojection,
                           gfx_batch::DepthShader* depthShader,
                           Flags flags,
                           const sensor::VisualSensor* visualSensor)
    : pimpl_(spimpl::make_unique_impl<Impl>(size,
                                            depthUnprojection,
                                            depthShader,
                                            flags,
                                            visualSensor)) {}

void RenderTarget::renderEnter() {
  pimpl_->renderEnter();
}

void RenderTarget::renderReEnter() {
  pimpl_->renderReEnter();
}

void RenderTarget::renderExit() {
  pimpl_->renderExit();
}

void RenderTarget::readFrameRgba(const Mn::MutableImageView2D& view) {
  pimpl_->readFrameRgba(view);
}

void RenderTarget::readFrameDepth(const Mn::MutableImageView2D& view) {
  pimpl_->readFrameDepth(view);
}

void RenderTarget::readFrameObjectId(const Mn::MutableImageView2D& view) {
  pimpl_->readFrameObjectId(view);
}

void RenderTarget::blitRgbaTo(Mn::GL::AbstractFramebuffer& target,
                              const Mn::Range2Di& targetRectangle) {
  pimpl_->blitRgbaTo(target, targetRectangle);
}

bool RenderTarget::blitRgbaTo(RenderTarget& target) {
  return pimpl_->blitRgbaTo(*target.pimpl_);
}

bool RenderTarget::blitDepthTo(RenderTarget& target) {
  return pimpl_->blitDepthTo(*target.pimpl_);
}

void RenderTarget::blitRgbaToDefault() {
  pimpl_->blitRgbaTo(Mn::GL::defaultFramebuffer,
                     Mn::GL::defaultFramebuffer.viewport());
}

Mn::Vector2i RenderTarget::framebufferSize() const {
  return pimpl_->framebufferSize();
}

Mn::GL::Texture2D& RenderTarget::getDepthTexture() {
  return pimpl_->getDepthTexture();
}

Mn::GL::Texture2D& RenderTarget::getObjectIdTexture() {
  return pimpl_->getObjectIdTexture();
}

Mn::GL::Texture2D& RenderTarget::getColorTexture() {
  return pimpl_->getColorTexture();
}

#ifdef ESP_BUILD_WITH_CUDA
Mn::GL::Texture2D& RenderTarget::getGaussianColorTexture() {
  return pimpl_->getGaussianColorTexture();
}

Mn::GL::Texture2D& RenderTarget::getCudaDepthTexture() {
  return pimpl_->getCudaDepthTexture();
}
#endif

void RenderTarget::tryDrawHbao() {
  return pimpl_->tryDrawHbao();
}

#ifdef ESP_BUILD_WITH_CUDA
void RenderTarget::compositeGaussian(GaussianSplattingShader& shader,
                                     bool hasColor) {
  pimpl_->compositeGaussian(shader, hasColor);
}
#endif

bool RenderTarget::hasColorAttachment() const {
  return pimpl_->hasColorAttachment();
}

bool RenderTarget::hadGaussianCompositeInLastPass() const {
  return pimpl_->hadGaussianCompositeInLastPass();
}

#ifdef ESP_BUILD_WITH_CUDA
void RenderTarget::readFrameRgbaGPU(uint8_t* devPtr) {
  pimpl_->readFrameRgbaGPU(devPtr);
}

void RenderTarget::readFrameDepthGPU(float* devPtr) {
  pimpl_->readFrameDepthGPU(devPtr);
}

void RenderTarget::readFrameObjectIdGPU(int32_t* devPtr) {
  pimpl_->readFrameObjectIdGPU(devPtr);
}
#endif

}  // namespace gfx
}  // namespace esp
