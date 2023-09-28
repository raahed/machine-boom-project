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

#ifndef AGXOSG_RENDERTOIMAGE_H
#define AGXOSG_RENDERTOIMAGE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!

#include <agxOSG/export.h>
#include <agxOSG/RenderTarget.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(RenderToImage);
  AGX_DECLARE_VECTOR_TYPES(RenderToImage);

  /**
  This class will transfer the camera buffer to the CPU for local storage/processing
  */
  class AGXOSG_EXPORT RenderToImage : public agxOSG::RenderTarget
  {
    public:

      /**
      Constructor of a RenderTarget that can render the camera buffer to an image.
      \param width - Width of the render target/image
      \param height - Height of the render target/image
      \param bufferComponent - Determines which component to render to the target, DEPTH_BUFFER or COLOR_BUFFER
      \param multiSamples - Specifies the number of multisamples to use when rendering this target
      \param textureFormat - Specifies the texture format of the render target
      */
      RenderToImage(agx::UInt width, agx::UInt height, BufferComponent bufferComponent = COLOR_BUFFER, agx::UInt multiSamples = 8, RenderTarget::TextureFormat textureFormat = RenderTarget::RGBA);

      /**
      \return pointer to the image data of this buffer
      */
      osg::Image* getImage();

      /**
      Will try to extract the image data as a raw data array of length \p size
      \param img - Image data as a raw data array
      \param size - The number of bytes that should be available in img. \p size must be equal to the size of the data in the image.
      \returns true if size==actual size of image data
      */
      bool extractImageData(unsigned char* img, int size);

      /// \return a pointer to the raw image data
      unsigned char* getImageData();

      /**
      Save the current image to disk given a filename
      Note: Only works for COLOR_BUFFER not DEPTH_BUFFER
      \param filename - Name of image file with specific suffix (.bmp, png, .jpg...)
      \return true if saving image to disk succeeded.
      */
      bool saveImage(const agx::String& filename);

    protected:
      virtual ~RenderToImage();

    private:
      osg::ref_ptr<osg::Image>            m_renderImage;
  };
}

#endif /* AGXOSG_RENDERTOTEXTURE_H */
