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

#ifndef AGXCOLLIDE_CONVEX_UTILS_H
#define AGXCOLLIDE_CONVEX_UTILS_H

#include <agxCollide/Contacts.h>
#include <agxCollide/ShapeCollider.h>

namespace agxCollide
{

  /**
  Collides two convex shapes (Box, Capsule, Convex, Cylinder, Sphere).
  \param convexShape1 The first convex shape.
  \param convexShape2 The second convex shape.
  \param transform1 The transformation matrix from the first convex shape's
  local frame to world frame.
  \param transform2 The transformation matrix from the second convex shape's
  local frame to world frame.
  \param result The contact point vector (probably non-empty from start)
  where resulting contact points get added.
  \param epaAccuracy Which accuracy does EPA have to reach in order to
  accept the found point as the deepest one?
  */


  AGXPHYSICS_EXPORT void collideConvexConvex(
    const Shape* convexShape1,
    const Shape* convexShape2,
    const agx::AffineMatrix4x4& transform1,
    const agx::AffineMatrix4x4& transform2,
    ShapeCollider::LocalContactPointVector& result,
    const agx::Real epaAccuracy = agx::AGX_EQUIVALENT_EPSILON);


  /**
  Given 2 convex shapes and 1 contact point (which is assumed to be the deepest possible point)
  this function will create extra contact points.
  \param shape1 The first convex shape.
  \param shape2 The second convex shape.
  \param transform1 The transformation matrix from the first convex shape's
  local frame to world frame.
  \param transform2 The transformation matrix from the second convex shape's
  local frame to world frame.
  \param point The first, deepest contact point.
  \param normal The contact normal belonging to the first, deepest contact point.
  \param deepestDepth The depth belonging to the first, deepest contact point.
  \param result The contact point vector (probably non-empty from start)
  where resulting contact points get added.
  \param keepOriginalPoint Result: Should the original point be kept, or have we found better ones?
  */
  AGXPHYSICS_EXPORT void createSupportContactPoints(
    const agxCollide::Shape* shape1,
    const agxCollide::Shape* shape2,
    const agx::AffineMatrix4x4& transform1,
    const agx::AffineMatrix4x4& transform2,
    const agx::Vec3& point,
    const agx::Vec3& normal,
    const agx::Real deepestDepth,
    agxCollide::ShapeCollider::LocalContactPointVector& result,
    bool& keepOriginalPoint );

  /**
  Check if a point in world coordinates is inside a given convex.
  \param point The point in world coordinates.
  \param convex The convex.
  */
  AGXPHYSICS_EXPORT bool isPointInsideConvex(const agx::Vec3& point, const agxCollide::Convex* convex);
}


#endif


