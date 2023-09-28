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

#include <agx/macros.h>
#include <agxCollide/ShapeCollider.h>
#include <agxCollide/ColliderTable.h>
#include <agx/Physics/Geometry/ShapeGroupEntity.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxCollide
{
  template <agx::UInt8 ShapeType>
  class GroupFooCollider : public ShapeCollider
  {
  public:
    GroupFooCollider() : ShapeCollider(agxCollide::Shape::GROUP, ShapeType) {}
    virtual ~GroupFooCollider() {}

    virtual void _calculateContacts(
        agx::Physics::Geometry::ShapeInstance shape1, agx::Physics::Geometry::ShapeInstance shape2,
        const agx::AffineMatrix4x4&, const agx::AffineMatrix4x4& transform2,
#if SHAPE_COLLIDER_VELOCITIES
        const agx::Vec3& linearVelocity1, const agx::Vec3& linearVelocity2,
#else
        const agx::Vec3& /*linearVelocity1*/, const agx::Vec3& linearVelocity2,
#endif
        const agx::Vec3& angularVelocity1, const agx::Vec3& angularVelocity2,
        bool /*earlyOut*/, LocalContactPointVector& result) override /// \todo Support earlyOut for GroupFooCollider.
    {
      agx::Physics::Geometry::ShapeGroupInstance group = shape1;


      const agx::Vector<agx::Physics::Geometry::ShapePtr>& childShapes = group.childShapes();
      // const agx::Vector<AffineMatrix4x4>& childTransforms = group.childTransforms();
      for (size_t i = 0; i < childShapes.size(); ++i)
      {
        agx::Physics::Geometry::ShapeInstance childShape = childShapes[i];

        ShapeCollider *collider = ColliderTable::getCollider(childShape.type(), shape2.type());

        // AffineMatrix4x4& localTransform = childTransforms[i];
        // AffineMatrix4x4 childTransform = localTransform * transform1;

        #if SHAPE_COLLIDER_VELOCITIES
        agx::Vec3 childLinearVelocity = (angularVelocity1 ^ localTransform.getTranslate()) + linearVelocity1;
        #else
        agx::Vec3 childLinearVelocity;
        #endif

        if ( !collider )
          continue;

        collider->calculateContacts(
            childShape, shape2,
            childShape.transform(), transform2,
            childLinearVelocity, linearVelocity2,
            angularVelocity1, angularVelocity2,
            false, result);
      }
    }
  };
}

DOXYGEN_END_INTERNAL_BLOCK()

