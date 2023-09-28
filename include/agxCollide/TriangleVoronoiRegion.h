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

#ifndef AGXCOLLIDE_TRIANGLEVORONOIREGION_H
#define AGXCOLLIDE_TRIANGLEVORONOIREGION_H

#include <agx/agx.h>
#include <agx/agxPhysics_export.h>
#include <agx/debug.h>

namespace agxCollide
{
  /**
  * Class containing the Voronoi region of a triangle as an uint8_t.
  * 0-2 stand for vertices, 3-5 for edges, and 6 for face.
  * The edges are numerated in the same way as the vertices they start from.
  */
  class AGXPHYSICS_EXPORT TriangleVoronoiRegion {
  public:
    /// The type of Voronoi region
    enum Type {
      VERTEX = 0,
      EDGE,
      FACE
    };

    /**
    * \param voronoiIndex  An uint8_t indicating the Triangle's
    *                      Voronoi Region. 0-2 vertices, 3-5 edges, 6 face.
    */
    TriangleVoronoiRegion( uint8_t voronoiIndex );

    /// \return The type of the Voronoi region
    Type getType() const;

    static Type calculateType(unsigned index);

    /**
    * \return The Voronoi index of the Voronoi region.
    *  0-2 stand for vertices, 3-5 for edges, and 6 for face.
    */
    uint8_t getVoronoiIndex() const;

    /**
    One way to create Voronoi regions is to set a flag for each edge that
    the point is in contact with.
    Maximum 2 flags can be set in a non-degenerate triangle.
    This gives the ordering
    Face: 0
    Edges: 1, 2 or 4
    Vertices: 3, 6 or 5.
    These two static variables can be used in order to set the edge flags
    and convert from an edged-flagged to the general indexing version.
    */
    static const uint8_t edgeFlags[4]; // last one for padding
    static const uint8_t edgeFlaggedToStandard[8]; // last one for padding

  private:
    /// Hide standard constructor
    TriangleVoronoiRegion();

    uint8_t m_voronoiIndex;
  };


  //implementation
  inline TriangleVoronoiRegion::TriangleVoronoiRegion( uint8_t voronoiIndex ) : m_voronoiIndex(voronoiIndex)
  {
    agxAssert( voronoiIndex <= 6);
  }

  inline TriangleVoronoiRegion::Type TriangleVoronoiRegion::calculateType(unsigned index)
  {
    return (TriangleVoronoiRegion::Type( index / 3));
  }

  inline TriangleVoronoiRegion::Type TriangleVoronoiRegion::getType() const
  {
    return TriangleVoronoiRegion::calculateType(m_voronoiIndex);
  }


  inline uint8_t TriangleVoronoiRegion::getVoronoiIndex() const
  {
    return m_voronoiIndex;
  }

}

#endif

