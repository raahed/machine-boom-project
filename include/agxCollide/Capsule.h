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

#ifndef AGXCOLLIDE_CAPSULE_H
#define AGXCOLLIDE_CAPSULE_H

#include <agx/agxPhysics_export.h>
#include <agx/Physics/Geometry/CapsuleEntity.h>
#include <agxCollide/Shape.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(Capsule);

  /**
  Implements a Capsule shape for geometric intersection tests.

  The capsule is oriented as the Y-axis with origin at center of mass.
  */
  class AGXPHYSICS_EXPORT Capsule : public Shape
  {
    public:
      /**
      Constructor

      \param radius Radius of capsule
      \param height Height of capsule
      */
      Capsule( const agx::Real radius = agx::Real(1), const agx::Real height = agx::Real(1) );

      /**
      Set the height of the capsule.
      \param height - The new height
      */
      void setHeight( const agx::Real height );

      /**
      Set the radius of the capsule.
      \param radius - The new radius
      */
      void setRadius( const agx::Real radius );

      /**
      Set the radius and height of the capsule.
      \param radius - The new radius
      \param height - The new height
      */
      void set( const agx::Real radius, const agx::Real height );

      /// \return the height of the radius
      agx::Real getHeight() const;

      /// \return the radius of the radius
      agx::Real getRadius() const;

      virtual Shape *clone() const override;
      virtual agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;
      virtual agx::Real getVolume() const override;
      virtual bool hasSupportFunction() const override;

      /**
      Returns the support point on face of the shape.
      \param supportDirection - Direction in shape coords where to find a support point
      */
      virtual agx::Vec3 getSupportPoint(const agx::Vec3& supportDirection) const override;
      virtual const BoundingAABB& updateBoundingVolume() override;
      virtual BoundingAABB calculateLocalBound() const override;
      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const agx::Real radius, const agx::Real height);

      DOXYGEN_START_INTERNAL_BLOCK()


      agx::Physics::Geometry::CapsulePtr getEntity() const;
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Capsule);
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      /// Destructor
      virtual ~Capsule( void );

      /// Internal constructor
      Capsule(const agx::Real radius, const agx::Real height, Shape::Type type, agx::Physics::Geometry::ShapePtr entity );
  };


  /* Implementation */
  inline agx::Physics::Geometry::CapsulePtr Capsule::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE agx::Real Capsule::getHeight() const
  {
    return getEntity().height();
  }

  AGX_FORCE_INLINE agx::Real Capsule::getRadius() const
  {
    return getEntity().radius();
  }

  AGX_FORCE_INLINE BoundingAABB Capsule::calculateBound(const agx::AffineMatrix4x4& transform, const agx::Real radius, const agx::Real height)
  {
    const agx::Vec3 p1 = agx::Vec3(0, height * agx::Real(0.5), 0) * transform;
    const agx::Vec3 p2 = agx::Vec3(0, height * agx::Real(-0.5), 0) * transform;

    BoundingAABB bound (agx::Vec3::componentMin(p1, p2) - agx::Vec3(radius), agx::Vec3::componentMax(p1, p2) + agx::Vec3(radius));
    return bound;
  }

  AGX_FORCE_INLINE bool Capsule::hasSupportFunction()  const
  {
    return true;
  }

}

#endif
