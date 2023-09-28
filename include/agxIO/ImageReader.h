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

#ifndef AGXIO_IMAGEREADER_H
#define AGXIO_IMAGEREADER_H


#include <agxIO/Image.h>
#include <agx/String.h>


namespace agxIO {

  /**
  Base class for ImageReaders.
  */
  class AGXPHYSICS_EXPORT ImageReader : public agx::Referenced {

    public:
      /**
      Reads an image from file.
      \return An Image, nullptr on error.
      */
      virtual Image* readImage( const agx::String& filename ) = 0;

  };

  typedef agx::ref_ptr<ImageReader> ImageReaderRef;
}

#endif

