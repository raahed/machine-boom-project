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

#include <agxCollide/ShapeCollider.h>
#include <agxCollide/ColliderTable.h>
#include <agxCollide/ConvexUtils.h>

namespace agxCollide
{

  /**
  This template class can be instantiated to use Convex-Convex for shape pairs where
  no explicit collider exists.
  The requirement for this to work is that:
  - Both shape types implementation of hasSupportFunction() returns true.
  - both shape types have getSupportPoint(const Vec3& dir) implmented

  */
  template <agx::UInt8 ShapeType1, agx::UInt8 ShapeType2>
  class FallbackConvexCollider : public ShapeCollider
  {
  public:
    FallbackConvexCollider() : ShapeCollider(ShapeType1, ShapeType2) {}
    virtual ~FallbackConvexCollider() {}

    virtual void _calculateContacts(
        agx::Physics::Geometry::ShapeInstance shape1, agx::Physics::Geometry::ShapeInstance shape2,
        const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
        const agx::Vec3& /*linearVelocity1*/, const agx::Vec3& /*linearVelocity2*/,
        const agx::Vec3& /*angularVelocity1*/, const agx::Vec3& /*angularVelocity2*/,
        bool /*earlyOut*/, LocalContactPointVector& result) override
    {
      const Shape* shape1ptr = shape1.model();
      const Shape* shape2ptr = shape2.model();

      if ( !shape1ptr->hasSupportFunction() || !shape2ptr->hasSupportFunction() )
        return;

      collideConvexConvex(shape1ptr, shape2ptr, transform1, transform2, result);
    }

  };

}


