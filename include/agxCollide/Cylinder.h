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

#ifndef AGXCOLLIDE_CYLINDER_H
#define AGXCOLLIDE_CYLINDER_H

#include <agx/agxPhysics_export.h>

#include <agx/agx.h>
#include <agx/Physics/Geometry/CylinderEntity.h>
#include <agxCollide/Shape.h>
#include <agxCollide/Capsule.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{

  AGX_DECLARE_POINTER_TYPES(Cylinder);
  /**
  A cylinder shape for geometric intersection tests.
  Definition a cylinder is oriented along the Y-axis with origin at center of mass.
  */
  class AGXPHYSICS_EXPORT Cylinder : public Shape
  {
    public:
      /**
      Constructor
      \param radius Radius of cylinder
      \param height Height of cylinder
      */
      Cylinder( const agx::Real radius, const agx::Real height );

      /**
      Set the height of a cylinder.
      \param height - New height of the cylinder
      */
      void setHeight( const agx::Real height );

      /**
      Set the radius of a cylinder.
      \param radius - New radius of the cylinder
      */
      void setRadius( const agx::Real radius );

      /**
      Set the radius and height of a cylinder.
      \param height - New height of the cylinder
      \param radius - New radius of the cylinder
      */
      void set( const agx::Real radius, const agx::Real height );

      /// \return the current height of the cylinder
      agx::Real getHeight() const;

      /// \return the current radius of the cylinder
      agx::Real getRadius() const;

      /**
      Calculate the inertia of this Cylinder
      \return the calculated inertia tensor for this Cylinder in the local coordinate system.
      */
      agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;

      /**
      Calculate the volume of this cylinder.
      \return the calculated volume for this cylinder.
      */
      virtual agx::Real getVolume() const override;

      virtual bool hasSupportFunction() const override;
      virtual Shape *clone() const override;

      virtual agx::Vec3 getSupportPoint( const agx::Vec3& supportDirection ) const override;
      virtual const BoundingAABB& updateBoundingVolume() override;
      virtual BoundingAABB calculateLocalBound() const override;
      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const agx::Real radius, const agx::Real height);

      DOXYGEN_START_INTERNAL_BLOCK()


      agx::Physics::Geometry::CylinderPtr getEntity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Cylinder);
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      /// Hidden default constructor
      Cylinder();

      /// Destructor
      virtual ~Cylinder();
  };

  /* Implementation */
  inline agx::Physics::Geometry::CylinderPtr Cylinder::getEntity() const
  {
    return m_entity;
  }
  AGX_FORCE_INLINE agx::Real Cylinder::getHeight() const
  {
    return getEntity().height();
  }
  AGX_FORCE_INLINE agx::Real Cylinder::getRadius() const
  {
    return getEntity().radius();
  }
  AGX_FORCE_INLINE agx::Real Cylinder::getVolume() const
  {
    return getRadius() * getRadius() * getHeight() * agx::PI;
  }

  AGX_FORCE_INLINE BoundingAABB Cylinder::calculateBound(const agx::AffineMatrix4x4& transform, const agx::Real radius, const agx::Real height)
  {
    return Capsule::calculateBound(transform, radius, height); // TODO The cylinder bound can be further optimized
  }

  inline bool Cylinder::hasSupportFunction() const
  {
    return true;
  }

}

#endif
