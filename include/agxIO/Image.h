/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXIO_IMAGE_H
#define AGXIO_IMAGE_H

#include <agx/agxPhysics_export.h>

#include <agx/Referenced.h>

namespace agxIO
{


  AGX_DECLARE_POINTER_TYPES(Image);
  /**
  Class to represent Images.
  */
  class AGXPHYSICS_EXPORT Image : public agx::Referenced
  {
    public:

      /// Specify row/column mode
      enum ImageDataMode {
        ROW_MAJOR,     //!< Pixel data is stored by row
        COLUMN_MAJOR   //!< Pixel data is stored by column
      };

      Image();

      /**
      \return Width of the image
      */
      unsigned int getWidth() const;

      /**
      \return Height of the image
      */
      unsigned int getHeight() const;

      /**
      \return number of channels in the image
      */
      unsigned int getChannels() const;

      /**
      \return Pointer to actual pixels values
      */
      unsigned char* getData();
      const unsigned char* getData() const;

      int getImageDataMode() const;

      /**
      Sets the image. The image takes responsibility for the data pointer.
      */
      void setImage(const unsigned int& width, const unsigned int& height,
                    const unsigned int& channels, const int& mode, unsigned char* data);
    protected:

      unsigned int m_width;
      unsigned int m_height;
      unsigned int m_channels;

      unsigned char* m_data;

      int m_imageDataMode;

      virtual ~Image();
  };

  typedef agx::ref_ptr<Image> ImageRef;
}

#endif

