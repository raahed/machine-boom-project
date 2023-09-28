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

#ifndef AGXCOLLIDE_POLYGONCLIPPING_H
#define AGXCOLLIDE_POLYGONCLIPPING_H

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#include <agx/agx_vector_types.h>
#include <agx/agxPhysics_export.h>
#include <agx/StackArray.h>
#include <agxCollide/BasicPrimitiveTests.h>

#include <agx/Integer.h>

#ifndef _WIN32
#include <float.h>
#endif

#include <agx/agx.h>
#include <agx/Vec3.h>

namespace agxCollide
{


  template <size_t N>
  bool clipConvexPolygonAgainstTrianglePrism(
    const agx::StackArray<agx::Vec3, N>& cPolygon,
    const agx::StackArray<agx::Vec3,3>& triangle,
    const agx::Vec3& normalTriangle,
    agx::StackArray<agx::Vec3, N + 3>& result,
    const agx::Real epsilon );


  /// Implementation


  template <size_t N>
  AGX_FORCE_INLINE bool clipConvexPolygonAgainstTrianglePrism(
    const agx::StackArray<agx::Vec3, N>& cPolygon,
    const agx::StackArray<agx::Vec3, 3>& triangle,
    const agx::Vec3& normalTriangle,
    agx::StackArray<agx::Vec3, N + 3>& result,
    const agx::Real epsilon )
  {
    //
    // Modified Sutherland-Hodgman polygon clipping in 3D.
    //
    if (cPolygon.size() == 0)
      return false;

    result.clear();

    for (size_t i = 0; i < cPolygon.size(); ++i)
      result.push_back(cPolygon[i]);

    agx::StackArray<agx::Vec3, 3> planeNormals; // these planeNormals are not normalized. Pointing outside from triangle.
    agx::StackArray<agx::Real, 3> planeDist;

    planeNormals.push_back((triangle[1] - triangle[0]) ^ normalTriangle);
    planeNormals.push_back((triangle[2] - triangle[1]) ^ normalTriangle);
    planeNormals.push_back((triangle[0] - triangle[2]) ^ normalTriangle);
    planeDist.push_back(triangle[0] * planeNormals[0]);
    planeDist.push_back(triangle[1] * planeNormals[1]);
    planeDist.push_back(triangle[2] * planeNormals[2]);


    agx::StackArray<agx::Vec3, N + 3> tmp;

    // clip convex polygon against each plane defined by triangle edges and normal
    for (unsigned int i = 0; i < 3; ++i) {
      tmp.clear();
      bool lastPointWasOutside = result[0] * planeNormals[i] - planeDist[i] > epsilon;
      // test each edge of the polygon
      for (size_t j = 0; j < result.size(); ++j) {
        size_t j_next = (j + 1) % result.size();
        bool nextPointIsOutside = result[j_next] * planeNormals[i] - planeDist[i] > epsilon;
        if (lastPointWasOutside != nextPointIsOutside) {
          // crossing plane of triangle, clip
          agx::Vec3 newPoint;
          agx::Real tmpT;
          if (intersectLineSegmentHyperPlane( result[j], result[j_next], planeNormals[i], planeDist[i], tmpT, newPoint, epsilon ))
            tmp.push_back( newPoint );
        }
        if (!nextPointIsOutside)
          tmp.push_back( result[j_next] );
        lastPointWasOutside = nextPointIsOutside;
      }

      result.clear();
      if (tmp.size() == 0)
        return false;

      for (size_t j = 0; j < tmp.size(); ++j) {
        if (result.size() == 0 ||
          ((result.back() - tmp[j]).length2() > epsilon && (result.front() - tmp[j]).length2() > epsilon))
          result.push_back( tmp[j] );
      }
    }
    return true;
  }

}

DOXYGEN_END_INTERNAL_BLOCK()
#endif
