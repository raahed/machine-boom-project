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

#include <agxSDK/MergeSplitThresholds.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( GeometryContactMergeSplitThresholds );

  /**
  Specific thresholds used to merge and split given geometry contacts.
  */
  class AGXPHYSICS_EXPORT GeometryContactMergeSplitThresholds : public MergeSplitThresholds
  {
    public:
      /**
      Indices to the threshold values when a body should be merged.
      */
      enum MergeThresholds : agx::UInt
      {
        MAX_RELATIVE_NORMAL_SPEED,  /**< If the speed along a contact normal is less than this
                                         value, the object may be merged. */
        MAX_RELATIVE_TANGENT_SPEED, /**< If the speed along a contact tangent is less than this
                                         value, the object may be merged. */
        MAX_ROLLING_SPEED,          /**< If the relative rolling speed is less than this the
                                         object may be merged. */
        NUM_MERGE_THRESHOLDS
      };

      /**
      Indices to the threshold values when a body should be split.
      */
      enum SplitThresholds : agx::UInt
      {
        MAX_IMPACT_SPEED = NUM_MERGE_THRESHOLDS, /**< If the impact speed (approaching each other) is
                                                      higher than this value, the objects may split. */
        LOGICAL_IMPACT,                          /**< Split when the state of the geometry contact is
                                                      agxCollide::GeometryContact::IMPACT_STATE. */
        MAY_SPLIT_IN_GRAVITY_FIELD,              /**< Check split given external forces for all objects
                                                      merged (i.e., rb->getForce() the sum of rb->addForce(),
                                                      including the gravity force). Performance warning,
                                                      disabled by default. */
        TANGENTIAL_ADHESION,                     /**< Adhesive force in the tangential directions preventing
                                                      the object to split (if > 0). Default: 0.0 */
        NORMAL_ADHESION,                         /**< Adhesive force in the normal direction preventing the
                                                      object to split (if > 0). Default: 0.0 */
        NUM_THRESHOLDS
      };

    public:
      /**
      Default constructor, all thresholds will be set to default.
      */
      GeometryContactMergeSplitThresholds();

      /**
      Assign maximum speed along a contact normal for a contact
      to be considered resting. Default: 0.01.
      */
      void setMaxRelativeNormalSpeed( agx::Real value );

      /**
      \param other - if given, the non-default/non-global value will
                      be used and if both are explicit, the maximum
                      value of the two is returned
      \return the maximum speed along a contact normal for a contact to
              be considered resting
      */
      agx::Real getMaxRelativeNormalSpeed( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      Assign maximum (sliding) speed along a contact tangent
      for a contact to be considered resting. Default 0.01.
      */
      void setMaxRelativeTangentSpeed( agx::Real value );

      /**
      \param other - if given, the non-default/non-global value will
                     be used and if both are explicit, the maximum
                     value of the two is returned
      \return the maximum (sliding) speed along a contact tangent for
              a contact to be considered resting
      */
      agx::Real getMaxRelativeTangentSpeed( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      Assign maximum (rolling) speed for a contact to be considered
      resting. Default 0.01.
      */
      void setMaxRollingSpeed( agx::Real value );

      /**
      \param other - if given, the non-default/non-global value will
                     be used and if both are explicit, the maximum
                     value of the two is returned
      \return the maximum (rolling) speed for a contact to be considered resting
      */
      agx::Real getMaxRollingSpeed( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      Assign maximum impact speed (along a contact normal) a merged
      object can resist without being split. Default 0.01.
      */
      void setMaxImpactSpeed( agx::Real value );

      /**
      \param other - if given, the non-default/non-global value will be
                     used and if both are explicit, the maximum value of
                     the two is returned
      \return the minimum impact speed (along the contact normal) for an
              object to split another object, i.e., the maximum impact speed
              a merged object can resist without being split
      */
      agx::Real getMaxImpactSpeed( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      True to split when geometry contact state is agxCollide::GeometryContact::IMPACT_STATE,
      i.e., the first time the objects collide. Default is false and "max impact speed" will
      be used instead.
      */
      void setSplitOnLogicalImpact( agx::Bool value );

      /**
      \param other - if given, the non-default/non-global value will be used and
                     if both are explicit, the maximum value (true) of the two is returned
      \return flag if a merged object will split on agxCollide::GeometryContact::IMPACT_STATE
              (i.e., first time the two objects collides)
      */
      agx::Bool getSplitOnLogicalImpact( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      Check split given external forces for all objects merged (i.e., rb->getForce()
      the sum of rb->addForce(), including the gravity force). Performance warning,
      disabled by default.
      \param maySplitInGravityField - true to enable, false to disable
      */
      void setMaySplitInGravityField( agx::Bool maySplitInGravityField );

      /**
      \param other - if given, the non-default/non-global flag will be used and if both
                     are explicit, the maximum value (true) of the two is returned
      \return true if split in gravity field is enabled - otherwise false
      */
      agx::Bool getMaySplitInGravityField( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      Adhesive force in the tangential directions preventing the object to split (if > 0)
      when the object is subject to external interactions (e.g., constraints). Default: 0.0
      \param value - tangential adhesive force
      */
      void setTangentialAdhesion( agx::Real value );

      /**
      \param other - if given, the non-default/non-global flag will be used and if both
                     are explicit, the maximum value of the two is returned
      \return the tangential adhesive force
      */
      agx::Real getTangentialAdhesion( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

      /**
      Adhesive force in the normal directions preventing the object to split (if > 0)
      when the object is subject to external interactions (e.g., constraints). Default: 0.0
      \param value - normal direction adhesive force
      */
      void setNormalAdhesion( agx::Real value );

      /**
      \param other - if given, the non-default/non-global flag will be used and if both
                     are explicit, the maximum value of the two is returned
      \return the normal direction adhesive force
      */
      agx::Real getNormalAdhesion( const GeometryContactMergeSplitThresholds* other = nullptr ) const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::GeometryContactMergeSplitThresholds );

      /**
      Creates and returns a clone of this object.
      */
      virtual agx::ICloneableRef clone( agx::ICloneable* child = nullptr ) override;

      /**
      Reset all values to default.
      */
      virtual void resetToDefault() override;

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~GeometryContactMergeSplitThresholds();
  };

  AGX_FORCE_INLINE agx::Real GeometryContactMergeSplitThresholds::getMaxRelativeNormalSpeed( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( MAX_RELATIVE_NORMAL_SPEED, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } );
  }

  AGX_FORCE_INLINE agx::Real GeometryContactMergeSplitThresholds::getMaxRelativeTangentSpeed( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( MAX_RELATIVE_TANGENT_SPEED, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } );
  }

  AGX_FORCE_INLINE agx::Real GeometryContactMergeSplitThresholds::getMaxRollingSpeed( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( MAX_ROLLING_SPEED, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } );
  }

  AGX_FORCE_INLINE agx::Real GeometryContactMergeSplitThresholds::getMaxImpactSpeed( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( MAX_IMPACT_SPEED, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } );
  }

  AGX_FORCE_INLINE agx::Bool GeometryContactMergeSplitThresholds::getSplitOnLogicalImpact( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( LOGICAL_IMPACT, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } ) > agx::Real( 0.5 );
  }

  AGX_FORCE_INLINE agx::Bool GeometryContactMergeSplitThresholds::getMaySplitInGravityField( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( MAY_SPLIT_IN_GRAVITY_FIELD, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } ) > agx::Real( 0.5 );
  }

  AGX_FORCE_INLINE agx::Real GeometryContactMergeSplitThresholds::getTangentialAdhesion( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( TANGENTIAL_ADHESION, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } );
  }

  AGX_FORCE_INLINE agx::Real GeometryContactMergeSplitThresholds::getNormalAdhesion( const GeometryContactMergeSplitThresholds* other ) const
  {
    return get( NORMAL_ADHESION, other, []( agx::Real a, agx::Real b ) -> agx::Real { return std::max( a, b ); } );
  }
}
