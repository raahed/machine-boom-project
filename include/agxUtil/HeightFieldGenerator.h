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

#ifndef AGXUTIL_HEIGHTFIELDGENERATOR_H
#define AGXUTIL_HEIGHTFIELDGENERATOR_H

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/String.h>

namespace agxCollide {
  class HeightField;
}

namespace agxIO {
  class Image;
}

namespace agxUtil {

  /// Class for generating a HeightField shape from an image
  AGX_DECLARE_POINTER_TYPES(HeightFieldGenerator);
  class AGXPHYSICS_EXPORT HeightFieldGenerator : public agx::Referenced {

    public:

      /// Default constructor
      HeightFieldGenerator() : Referenced() {}

      /**
      \param image - Pointer to an image used for creating the height field
      \param sizeX - World size in the x dimension
      \param sizeY - World size in the y dimension
      \param low   - The lowest height in the z dimension, maps to the lowest value in the image
      \param high  - The highest height in the z dimension, maps to the highest value in the image
      \param bottomMargin How deep is the HeightField under its lowest point?
      \return a HeightField shape with extents in X/Y and height values along Z
      */
      static agxCollide::HeightField* createHeightFieldFromImage( agxIO::Image* image,
                                                                  agx::Real sizeX, agx::Real sizeY,
                                                                  agx::Real low, agx::Real high,
                                                                  agx::Real bottomMargin=agx::Real(1));

      /**
      \param filename - Path to an image which will be read as a height map.
      \param sizeX - World size in the x dimension
      \param sizeY - World size in the y dimension
      \param low   - The lowest height in the z dimension, maps to the lowest value in the image
      \param high  - The highest height in the z dimension, maps to the highest value in the image
      \param bottomMargin How deep is the HeightField under its lowest point?
      \return a HeightField shape with extents in X/Y and height values along Z
      */
      static agxCollide::HeightField* createHeightFieldFromFile( const agx::String& filename,
                                                                 agx::Real sizeX, agx::Real sizeY,
                                                                 agx::Real low, agx::Real high,
                                                                 agx::Real bottomMargin=agx::Real(1));

    protected:
      virtual ~HeightFieldGenerator();

  };

}


#endif

