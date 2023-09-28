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

#ifndef AGXCOLLIDE_BOX_H
#define AGXCOLLIDE_BOX_H

#include <agxCollide/Shape.h>
#include <agx/Physics/Geometry/BoxEntity.h>


namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(Box);
  /**
  A box shape for geometric intersection tests.
  Origin is in center of box.
  The size is specified as half extents.
  */
  class AGXPHYSICS_EXPORT Box : public Shape
  {
    public:
      /**
      Construct given half extents.
      \param halfExtents - half size of box sides
      */
      Box( const agx::Vec3 halfExtents );

      /**
      Construct given half extents.
      \param halfExtentX - half size of box in x direction
      \param halfExtentY - half size of box in y direction
      \param halfExtentZ - half size of box in z direction
      */
      Box( agx::Real halfExtentX, agx::Real halfExtentY, agx::Real halfExtentZ );


      /**
      Set the half size of the box
      \param halfExtents - half size of box sides
      */
      void setHalfExtents( const agx::Vec3& halfExtents );

      /**
      \return half size of box sides
      */
      const agx::Vec3& getHalfExtents() const;

      virtual Shape *clone() const override;
      virtual agx::Real getVolume() const override;
      virtual agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;
      virtual bool hasSupportFunction() const override;
      virtual agx::Vec3 getSupportPoint( const agx::Vec3& supportDirection ) const override;
      virtual const BoundingAABB& updateBoundingVolume() override;
      virtual BoundingAABB calculateLocalBound() const override;

      /**
      Method for calculating the bound of a transformed box.
      \param transform - local to world transform
      \param halfExtents - Half extents of the box
      \return calculated bounding volume (AABB).
      */
      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const agx::Vec3& halfExtents);

      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Box);

      agx::Physics::Geometry::BoxPtr getEntity() const;

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      Box();

      virtual ~Box();
  };

  /* Implementation */
  inline const agx::Vec3& Box::getHalfExtents() const
  {
    return getEntity().halfExtents();
  }

  inline agx::Physics::Geometry::BoxPtr Box::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE BoundingAABB Box::calculateBound(const agx::AffineMatrix4x4& transform, const agx::Vec3& halfExtents)
  {
    return BoundingAABB(BoundingAABB(-halfExtents, halfExtents), transform);
  }
}


#endif
