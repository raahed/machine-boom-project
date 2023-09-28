/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agxOSG/export.h>
#include <agxOSG/ImageCapture.h>
#include <agx/Object.h>
#include <agx/Vec2.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Node>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Image>
#include <osg/Camera>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(RenderTarget);
  AGX_DECLARE_VECTOR_TYPES(RenderTarget);

  /**
  Render target is used for rendering camera content to a buffer, either a DEPTH or COLOR target
  */
  class AGXOSG_EXPORT RenderTarget : public agx::Object
  {
  public:


    /// Specifies which type of buffer to render
    enum BufferComponent
    {
      COLOR_BUFFER=osg::Camera::COLOR_BUFFER,
      DEPTH_BUFFER=osg::Camera::DEPTH_BUFFER
    };

    /// Specifies target texture format
    enum TextureFormat
    {
      RGBA=GL_RGBA,
      RGB = GL_RGB,
      DEPTH_COMPONENT= GL_DEPTH_COMPONENT
    };

    enum ComputeNearFarMode
    {
      DO_NOT_COMPUTE_NEAR_FAR=osg::Camera::DO_NOT_COMPUTE_NEAR_FAR,
      COMPUTE_NEAR_FAR_USING_BOUNDING_VOLUMES=osg::Camera::COMPUTE_NEAR_FAR_USING_BOUNDING_VOLUMES,
      COMPUTE_NEAR_FAR_USING_PRIMITIVES=osg::Camera::COMPUTE_NEAR_FAR_USING_PRIMITIVES,
      COMPUTE_NEAR_USING_PRIMITIVES=osg::Camera::COMPUTE_NEAR_USING_PRIMITIVES
    };

    /**
    Constructor. Create a new RenderTarget
    \param width - Width of target buffer
    \param height - Height of target buffer
    */
    RenderTarget(agx::UInt width, agx::UInt height);

    /**
    By using a reference camera, the render target camera will get the same view/projection/viewpoint as the reference.
    \param referenceCamera - The camera to mimic
    */
    void setReferenceCamera(const osg::Camera* referenceCamera);

    /**
    By attaching an ImageCaptureBase you can automatically store the image data to disk given a specific format, fps, path etc.
    \param imageCapture - Pointer to a ImageCaptureBase object that manages storing of images to disk
    */
    void setImageCapture(agxOSG::ImageCaptureBase *imageCapture);

    /**
    \return a pointer to the image capture object that this RenderTarget is using (nullptr if none)
    */
    agxOSG::ImageCaptureBase* getImageCapture();

    /**
    Update the size of the render target buffer size
    \param width - Width of render target
    \param height - Height of render target
    */
    void setSize(agx::UInt width, agx::UInt height);

    /**
    Update the size of the render target buffer size
    \param size - Width, height of render target
    */
    void setSize(agx::Vec2u size);

    /// \return the size of the render target
    agx::Vec2u getSize() const;

    /**
    \return the camera that is used for this RenderTarget
    */
    osg::Camera* getCamera();

    /**
    Add a pre-render callback to this RenderTarget
    \param callback - DrawCallback object to add
    */
    void addPreRenderCallback(osg::Camera::DrawCallback* callback);

    /**
    Update the view matrix given the arguments
    \param eye - position of eye
    \param center - point towards the camera is aimed
    \param up - up vector
    */
    void setViewMatrixAsLookAt(const agx::Vec3& eye, const agx::Vec3& center, const agx::Vec3& up);

    /**
    Update the perspective projection matrix given the arguments
    \param fovy - field of view in y
    \param aspectRatio - width/height ratio
    \param zNear - Near clipping plane
    \param zFar - Far clipping plane
    */
    void setProjectionMatrixAsPerspective(double fovy, double aspectRatio, double zNear, double zFar);

    /**
    Update the orthographic projection matrix given the arguments
    \param left, right, bottom, top - defines the size of the orthographic view frustum
    \param zNear - Near clipping plane
    \param zFar - Far clipping plane
    */
    void setProjectionMatrixAsOrtho(double left, double right, double bottom, double top, double zNear, double zFar);

    /**
    Specifies whether the near far should be automatically updated or not
    */
    void setComputeNearFarMode(agxOSG::RenderTarget::ComputeNearFarMode mode);

    /**
    Transform the position into screen coordinates using the camera associated to this RenderTarget
    \param worldPos - Position in world coordinate system
    \returns position (x,y) in screen coordinates
    */
    agx::Vec2 calculateScreenCoords(agx::Vec3& worldPos);

    /**
    Set the cull node mask to specify what part of the scenegraph that should be rendered.
    This maps directly to OSG:s setCullMask. 
    Typical use for this mask is: 

    // Disable mask:
    cullMask &= ~agxOSG::ExampleApplication::HUD_MASK;
    setCameraCullMask(cullMask);

    // Enable mask
    cullMask |= agxOSG::ExampleApplication::HUD_MASK;
    setCameraCullMask(cullMask);
    */
    void setCameraCullMask(int cullMask);

    /// \return current cullmask for the camera node
    int getCameraCullMask() const;


  protected:
    virtual ~RenderTarget();

  private:
    class CameraSynchronizeCallback;

  protected:
    agx::UInt m_width;
    agx::UInt m_height;
    ImageCaptureBaseRef                     m_imageCapture;
    osg::ref_ptr<osg::Camera>               m_camera;
    osg::ref_ptr<CameraSynchronizeCallback> m_cameraSynchronizeCallback;
  };


  DOXYGEN_START_INTERNAL_BLOCK()
  class RenderTarget::CameraSynchronizeCallback : public osg::Camera::DrawCallback
  {
  public:
    CameraSynchronizeCallback(const osg::Camera *reference, osg::Camera *target);
    virtual ~CameraSynchronizeCallback();

    virtual void operator () (osg::RenderInfo& /* renderInfo */) const;
    using osg::Camera::DrawCallback::operator();

    void addNestedCallback(osg::Camera::DrawCallback *child);

  private:
    osg::ref_ptr<const osg::Camera> m_reference;
    osg::ref_ptr<osg::Camera>       m_target;
    agx::Vector< osg::ref_ptr<osg::Camera::DrawCallback> > m_children;
  };
  DOXYGEN_END_INTERNAL_BLOCK()
}
