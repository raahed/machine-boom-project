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

#ifndef AGXIO_PNGREADER_H
#define AGXIO_PNGREADER_H

#include <agxIO/ImageReader.h>

namespace agxIO {

  /**
  ImageReader for png-files.
  */
  class AGXPHYSICS_EXPORT ImageReaderPNG : public agxIO::ImageReader {

    public:

      /// Constructor
      ImageReaderPNG() : ImageReader() {}

      /**
      Read an image from the specified filename. Memory for Image must be deallocated by user of class.
      \ return a pointer to the read image data if successful. nullptr if unable to read file
      */
      Image* readImage( const agx::String& filename );

    protected:
      virtual ~ImageReaderPNG() {}



  };

  typedef agx::ref_ptr<ImageReaderPNG> ImageReaderPNGRef;
}

#endif

