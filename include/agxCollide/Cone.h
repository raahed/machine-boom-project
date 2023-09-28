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
#include <agx/Physics/Geometry/ConeEntity.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{

  AGX_DECLARE_POINTER_TYPES(Cone);

  /**
  A right circular cone shape for geometric intersection tests.

  See manual for information about limitations and which shape pairs that are supported for cones.

  The cone can be configured as an orinary cone (topRadius==0) or a
  truncated Cone (topRadius != 0 ). The shape is defined along the Y-axis with the base described
  by the bottom radius in the XZ-plane at Y=0. The top of the cone, described by topRadius, is positioned
  at height along the positive Y axis.

  Requirements for the cone parameters are:
  - height > 0
  - topRadius >= 0
  - bottomRadius > topRadius

  If the topRadius is larger than 0, then the top part of the cone is truncated.
  */
  class AGXPHYSICS_EXPORT Cone : public Shape
  {
    public:

      /**
      Creates a cone.
      */
      Cone( agx::Real topRadius, agx::Real bottomRadius, agx::Real height );

      /**
      \return The top radius for the cone
      */
      agx::Real getTopRadius() const;

      /**
      \return The bottom radius for the cone
      */
      agx::Real getBottomRadius() const;

      /**
      \return The height of the cone
      */
      agx::Real getHeight() const;

      /**
      \return The height of the truncated part. This value is non-zero if topRadius is > 0.
      */
      agx::Real getTruncatedHeight() const;


      /**
      \return The angle between the cone axis and side. This value is the same as half the opening angle.
      */
      agx::Real getAngle() const;

      /**
      Sets all parameters for the cone.
      \param topRadius - New top radius
      \param bottomRadius - New bottom radius.
      \param height - New height for the cone.
      */
      void set( agx::Real topRadius, agx::Real bottomRadius, agx::Real height );

      /**
      Sets the top radius for the cone.
      */
      void setTopRadius( agx::Real r );

      /**
      Sets the bottom radius for the cone.
      */
      void setBottomRadius( agx::Real r );

      /**
      Sets the height of the cone.
      */
      void setHeight( agx::Real height );

      /**
      Calculate the inertia of the Cone.
      \return the calculated inertia tensor for this Cone in the local coordinate system.
      */
      agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;


      virtual agx::Vec3 getCenter() const override;

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


      DOXYGEN_START_INTERNAL_BLOCK()
      agx::Physics::Geometry::ConePtr getEntity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Cone);
      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      Compute the height of the truncated part of the cone.
      */
      static agx::Real computeTruncatedHeight( agx::Real topRadius, agx::Real bottomRadius, agx::Real height );

    protected:

      /// Hidden default constructor
      Cone();

      /// Destructor
      virtual ~Cone();
  };





  inline agx::Physics::Geometry::ConePtr Cone::getEntity() const
  {
    return m_entity;
  }


  inline agx::Real Cone::getTopRadius() const
  {
    return getEntity().topRadius();
  }

  inline agx::Real Cone::getBottomRadius() const
  {
    return getEntity().bottomRadius();
  }

  inline agx::Real Cone::getHeight() const
  {
    return getEntity().height();
  }


}
