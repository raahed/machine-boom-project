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

#include <agx/agx.h>
#include <agxCollide/Shape.h>
#include <agx/Physics/Geometry/HollowCylinderEntity.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{

  AGX_DECLARE_POINTER_TYPES(HollowCylinder);

  /**
  A hollow cylinder shape for geometric intersection tests.

  See manual for information about limitations and which shape pairs that are supported for hollow cylinders.

  The hollow cylinder axis is along the Y-axis. The shape center is at (0,0,0).

  Valid parameters for the hollow cylinder are:
  - height > 0
  - inner radius > 0
  - thickness > 0

  */
  class AGXPHYSICS_EXPORT HollowCylinder : public Shape
  {
    public:

      /**
      Creates a hollow cylinder.
      */
      HollowCylinder( agx::Real innerRadius, agx::Real height, agx::Real thickness );

      /**
      \return The outer radius for the hollow cylinder
      */
      agx::Real getOuterRadius() const;

      /**
      \return The inner radius for the hollow cylinder
      */
      agx::Real getInnerRadius() const;

      /**
      \return The height for the hollow cylinder
      */
      agx::Real getHeight() const;

      /**
      \return The thickness for the hollow cylinder
      */
      agx::Real getThickness() const;


      /**
      Sets all parameters for the hollow cylinder.
      \param radius - The new outer radius
      \param height - The new height
      \param thickness - The new thickness
      */
      void set( agx::Real radius, agx::Real height, agx::Real thickness );

      /**
      Sets the outer radius for the hollow cylinder.
      */
      void setOuterRadius( agx::Real r );

      /**
      Sets the inner radius for the hollow cylinder
      */
      void setInnerRadius( agx::Real r );

      /**
      Sets the height for the hollow cylinder
      */
      void setHeight( agx::Real h );

      /**
      Sets the thickness for the hollow cylinder.
      Updating the thickness will keep the inner hole size constant
      and the outer radius size will change.
      */
      void setThickness( agx::Real t );

      /**
      Calculate the inertia of the HollowCylinder.
      \return the calculated inertia tensor for this HollowCylinder in the local coordinate system.
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

      // static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const agx::Real bottomRadius, const agx::Real height);

      DOXYGEN_START_INTERNAL_BLOCK()
      agx::Physics::Geometry::HollowCylinderPtr getEntity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::HollowCylinder);
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      /// Hidden default constructor
      HollowCylinder();

      /// Destructor
      virtual ~HollowCylinder();
  };





  inline agx::Physics::Geometry::HollowCylinderPtr HollowCylinder::getEntity() const
  {
    return m_entity;
  }


  inline agx::Real HollowCylinder::getOuterRadius() const
  {
    return getEntity().radius();
  }

  inline agx::Real HollowCylinder::getInnerRadius() const
  {
    return getOuterRadius() - getThickness();
  }

  inline agx::Real HollowCylinder::getHeight() const
  {
    return getEntity().height();
  }


  inline agx::Real HollowCylinder::getThickness() const
  {
    return std::min( getEntity().radius(), getEntity().thickness() );
  }
}


