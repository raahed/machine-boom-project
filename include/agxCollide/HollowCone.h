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
#include <agx/Physics/Geometry/HollowConeEntity.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{

  AGX_DECLARE_POINTER_TYPES(HollowCone);

  /**
  A right circular hollow cone shape for geometric intersection tests.

  See manual for information about limitations and which shape pairs that are supported.

  The cone can be configured as an orinary cone (topRadius==0) or truncated  (topRadius != 0 ).
  The shape is defined along the Y-axis with the base described by the bottom radius in the
  XZ-plane at Y=0. The top of the cone, described by topRadius, is positioned
  at height along the positive Y axis.

  There are two ways to create a HollowCone, either via the five argument
  constructor that uses inside dimensions OR the four argument constructor
  that uses outside measurements.
  */
  class AGXPHYSICS_EXPORT HollowCone : public Shape
  {
    public:
      /**
      Creates a Hollow Cone based upon inner radiuses (hole size) and thickness.
      The hollow cone has constant thickness. That means that if the outerHeight is
      set to a to large value, it will be clamped to maintain constant thickness.

      If the HollowCone is truncated and the inside top radius is larger than 0, then
      innerHeight and outerHeight must have the same value. Otherwise the value for
      topInneRadius would not be given at the top.

      Requirements for the parameters:
        topInnerRadius >= 0
        bottomInnerRadius > topInnerRadius
        innerHeight > 0
        outerHeight >= innerHeight
        thickness > 0
      */
      HollowCone( agx::Real topInnerRadius, agx::Real bottomInnerRadius, agx::Real innerHeight, agx::Real outerHeight, agx::Real thickness );


      /**
      Creates a Hollow Cone based upon outside cone measurements and thickness to
      get a hollow inside.

      If topOuterRadius > 0, then the HollowCone is truncated.
      If topOuterRadius < thickness, then the hole does not pass through the shape

      Requirements for the parameters:
        topOuterRadius >= 0
        bottomOuterRadius > topOuterRadius
        outerHeight > 0
        thickness < bottomOuterRadius
      */
      HollowCone( agx::Real topOuterRadius, agx::Real bottomOuterRadius, agx::Real outerHeight, agx::Real thickness );


      /**
      \return The top outer radius for the hollow cone
      */
      agx::Real getTopOuterRadius() const;

      /**
      \return The bottom outer radius for the hollow cone
      */
      agx::Real getBottomOuterRadius() const;

      /**
      \return The bottom inner radius for the hollow cone
      */
      agx::Real getBottomInnerRadius() const;

      /**
      \return The hollow cone height
      */
      agx::Real getHeight() const;

      /**
      \return The the thickenss of the hollow cone.
      */
      agx::Real getThickness() const;

      /**
      Returns the height of the truncated part.
      This value is non-zero if topRadius is > 0.
      */
      agx::Real getTruncatedHeight() const;


      /**
      Returns the angle between the cone axis and side.
      This value is the same as half the opening angle.
      */
      agx::Real getAngle() const;

      /**
      Sets all parameters for the hollow cone.
      \param topOutsideRadius - New top radius
      \param bottomOutsideRadius - New bottom radius.
      \param height - New height for the hollow cone.
      \param thickness - New thickness for the hollow cone
      */
      void set( agx::Real topOutsideRadius, agx::Real bottomOutsideRadius, agx::Real height, agx::Real thickness );

      /**
      Sets the top radius for the hollow cone.
      */
      void setTopOuterRadius( agx::Real r );

      /**
      Sets the bottom radius for the hollow cone.
      */
      void setBottomOuterRadius( agx::Real r );

      /**
      Sets the height for the hollow cone.
      */
      void setHeight( agx::Real height );

      /**
      Sets the thickness for the hollow cone.
      */
      void setThickness( agx::Real thickness );

      /**
      Calculate the inertia of the HollowCone.
      \return the calculated inertia tensor for this HollowCone in the local coordinate system.
      */
      agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;

      /**
      Calculate the volume of this cylinder.
      \return the calculated volume for this cylinder.
      */
      virtual agx::Real getVolume() const override;

      virtual agx::Vec3 getCenter() const override;

      virtual bool hasSupportFunction() const override;

      virtual Shape *clone() const override;

      virtual agx::Vec3 getSupportPoint( const agx::Vec3& supportDirection ) const override;

      virtual const BoundingAABB& updateBoundingVolume() override;

      virtual BoundingAABB calculateLocalBound() const override;

      DOXYGEN_START_INTERNAL_BLOCK()
      agx::Physics::Geometry::HollowConePtr getEntity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::HollowCone);
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      /// Hidden default constructor
      HollowCone();

      /// Destructor
      virtual ~HollowCone();
  };





  inline agx::Physics::Geometry::HollowConePtr HollowCone::getEntity() const
  {
    return m_entity;
  }


  inline agx::Real HollowCone::getTopOuterRadius() const
  {
    return getEntity().topRadius();
  }

  inline agx::Real HollowCone::getBottomOuterRadius() const
  {
    return getEntity().bottomRadius();
  }


  inline agx::Real HollowCone::getBottomInnerRadius() const
  {
    return getBottomOuterRadius() - getThickness();
  }

  inline agx::Real HollowCone::getHeight() const
  {
    return getEntity().height();
  }

  inline agx::Real HollowCone::getThickness() const
  {
    return std::min( getEntity().bottomRadius(), getEntity().thickness() );
  }

}
