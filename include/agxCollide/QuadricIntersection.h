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

#pragma once

#include <agx/agxPhysics_export.h>
#include <agxCollide/Quadric.h>

#include <agx/StackArray.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide {

namespace QuadricIntersection {


  typedef agx::StackArray< agx::Vec2, 2 > Vec2Array2;
  typedef agx::StackArray< agx::Vec2, 4 > Vec2Array4;

  /**
  Split a degenerate conic into line(s)
  \param conic - Matrix describing conic
  \param line1 - output Vec3(a,b,c) describing line ax+by+c=0.
  \param line2 - output Vec3(a,b,c) describing line ax+by+c=0.
  */
  AGXPHYSICS_EXPORT bool splitConic( const agx::Matrix3x3& conic, agx::Vec3& line1, agx::Vec3& line2 );


  /**
  Test for line / conic intersection
  \param conic  - Matrix describing conic
  \param line   - Vec3(a,b,c) describing line ax+by+c=0
  \param point1 - Output point in homogeneous coords
  \param point2 - Output point in homogeneous coords
  */
  AGXPHYSICS_EXPORT bool lineConic(  const agx::Matrix3x3& conic, agx::Vec3 line, agx::Vec3& point1, agx::Vec3& point2 );

  /**
  Compute if quadrics qA (Any) and qC (Circle) overlaps.
  */
  AGXPHYSICS_EXPORT agx::UInt8 intersect( const Quadric2D& qA, const Quadric2D& qC, Vec2Array4& output );


  /**
  Computes and returns matrix Ms adjugate, the transpose of its cofactor matrix
  */
  AGXPHYSICS_EXPORT agx::Matrix3x3 adjugate( const agx::Matrix3x3& M );



}


}

DOXYGEN_END_INTERNAL_BLOCK()

