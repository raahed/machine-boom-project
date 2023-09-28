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

#ifndef AGXCOLLIDE_WIRE_SHAPE_H
#define AGXCOLLIDE_WIRE_SHAPE_H

#include <agx/agxPhysics_export.h>
#include <agx/Physics/Geometry/WireShapeEntity.h>
#include <agxCollide/Shape.h>
#include <agxCollide/Capsule.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(WireShape);

  /**
  Internal class used by agxWire::Wire.

  Implements a WireShape shape for geometric intersection tests.
  The WireShape is identical to a capsule for static collision detection,
  but adds the possibility to do continuous collision detection based on the
  linear movement of the capsule axis end points (the movement does not
  have to be identical, so that the capsule can deform during the time step).
  These end points have to be set newly each time step in order to be used by
  collision detection. Check the user manual in order to see which colliders
  can make use of continuous collision detection for WireShapes.

  The capsule is oriented as the Y-axis with origin at center of mass.
  */
  class AGXPHYSICS_EXPORT WireShape : public Capsule
  {
    public:
      /**
      Constructor
      \param radius Radius of capsule
      \param height Height of capsule
      \param previousEndPoint0 The previous first point in world coordinates.
      \param previousEndPoint1 The previous second point in world coordinates.
      */
      WireShape(
        const agx::Real radius, const agx::Real height,
        const agx::Vec3 previousEndPoint0 = agx::Vec3(),
        const agx::Vec3 previousEndPoint1 = agx::Vec3());

      /**
      \return the first previous end point (corresponding to (0, -h/2, 0)) in world coordinates.
      */
      agx::Vec3 getPreviousEndPoint0() const;

      /**
      \return the second previous end point (corresponding to (0, h/2, 0)) in world coordinates.
      */
      agx::Vec3 getPreviousEndPoint1() const;

      /*
      Sets radius, height, and the first previous end points for continuous collision detection.
      \param worldTransform New world transform of this wire shape.
      \param radius The new radius.
      \param height The new height.
      \param previousEndPoint0 Corresponding to local Vec3(0, -0.5h, 0), but in world coordinates.
      \param previousEndPoint1 Corresponding to local Vec3(0, 0.5h, 0), but in world coordinates.
      */
      agxCollide::BoundingAABB set( const agx::AffineMatrix4x4& worldTransform, const agx::Real radius, const agx::Real height, const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1 );

      virtual const BoundingAABB& updateBoundingVolume() override;
      virtual BoundingAABB calculateLocalBound() const override;
      static BoundingAABB calculateBound( const agx::AffineMatrix4x4& transform, const agx::Real radius, const agx::Real height, const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1 );

      agx::Physics::Geometry::WireShapePtr getEntity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::WireShape );

    protected:
      WireShape();
      virtual ~WireShape();
  };

  /* Implementation */
  inline agx::Physics::Geometry::WireShapePtr WireShape::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE agx::Vec3 WireShape::getPreviousEndPoint0() const
  {
    return getEntity().previousPoint0();
  }

  AGX_FORCE_INLINE agx::Vec3 WireShape::getPreviousEndPoint1() const
  {
    return getEntity().previousPoint1();
  }

  AGX_FORCE_INLINE BoundingAABB WireShape::calculateBound( const agx::AffineMatrix4x4& transform, const agx::Real radius, const agx::Real height, const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1 )
  {
    const agx::Vec3  p1  = agx::Vec3( 0, height * agx::Real(  0.5 ), 0 ) * transform;
    const agx::Vec3  p2  = agx::Vec3( 0, height * agx::Real( -0.5 ), 0 ) * transform;
    const agx::Vec3& p3  = previousEndPoint0;
    const agx::Vec3& p4  = previousEndPoint1;
    const agx::Vec3 rVec( radius );

    BoundingAABB bound( agx::Vec3::componentMin( agx::Vec3::componentMin( p1, p2 ), agx::Vec3::componentMin( p3, p4 ) ) - rVec,
                        agx::Vec3::componentMax( agx::Vec3::componentMax( p1, p2 ), agx::Vec3::componentMax( p3, p4 ) ) + rVec );
    return bound;
  }

}

#endif
