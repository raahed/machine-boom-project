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

#ifndef AGXCOLLIDE_SPHERE_H
#define AGXCOLLIDE_SPHERE_H

#include <agxCollide/Shape.h>
#include <agx/Physics/Geometry/SphereEntity.h>


namespace agxCollide
{

  AGX_DECLARE_POINTER_TYPES(Sphere);
  /**
  A sphere class for geometric intersection tests.
  */
  class AGXPHYSICS_EXPORT Sphere : public Shape
  {
    public:
      /**
      Constructor
      \param radius Radius of sphere
      */
      Sphere( agx::Real radius = agx::Real(0.5) );


      /**
      Set the sphere \p radius.
      */
      void setRadius( agx::Real radius );

      /**
      \return The sphere radius.
      */
      agx::Real getRadius() const;

      /* Common shape methods */
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
      virtual agx::Real calculateBoundingRadius( BoundingAABB& localAABB ) const override;

      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, agx::Real radius);

      static agx::Real calculateVolume(agx::Real radius);

      DOXYGEN_START_INTERNAL_BLOCK()


      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Sphere);
      agx::Physics::Geometry::SpherePtr getEntity() const;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /// Destructor
      virtual ~Sphere( void );

  };


  /* Implementation */
  AGX_FORCE_INLINE agx::Real Sphere::getRadius() const
  {
    return getEntity().radius();
  }
  inline agx::Physics::Geometry::SpherePtr Sphere::getEntity() const
  {
    return m_entity;
  }
  AGX_FORCE_INLINE BoundingAABB Sphere::calculateBound(const agx::AffineMatrix4x4& transform, agx::Real radius)
  {
    const agx::Vec3 pos = transform.getTranslate();
    const agx::Vec3 halfExtents = agx::Vec3(radius);
    const BoundingAABB bound = BoundingAABB(pos - halfExtents, pos + halfExtents);
    return bound;
  }

  AGX_FORCE_INLINE agx::Real Sphere::calculateVolume(agx::Real radius)
  {
    // 4 pi r^3 / 3
    return radius * radius * radius * (agx::Real(4) * agx::PI / agx::Real(3));
  }

}

#endif
