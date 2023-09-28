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
  AGX_DECLARE_POINTER_TYPES( ConstraintMergeSplitThresholds );

  /**
  Specific thresholds used to split and merge objects given constraints.
  */
  class AGXPHYSICS_EXPORT ConstraintMergeSplitThresholds : public MergeSplitThresholds
  {
    public:
      enum MergeThresholds : agx::UInt
      {
        MAX_RELATIVE_SPEED,
        NUM_MERGE_THRESHOLDS
      };

      enum SplitThresholds : agx::UInt
      {
        DESIRED_SPEED_DIFF = NUM_MERGE_THRESHOLDS,
        DESIRED_LOCK_ANGLE_DIFF,
        DESIRED_RANGE_ANGLE_DIFF,
        DESIRED_FORCE_RANGE_DIFF,
        NUM_THRESHOLDS
      };

    public:
      /**
      Default constructor, all thresholds will be set to default.
      */
      ConstraintMergeSplitThresholds();

      /**
      Assign maximum relative speed between the constrained objects for the system to be considered at rest.
      I.e., when the relative motion between the objects is less than this threshold, the objects may merge.
      Default: 0.005.
      */
      void setMaxRelativeSpeed( agx::Real value );

      /**
      When the relative motion between the objects is less than this threshold, the objects may merge.
      \return the maximum relative speed between the constrained objects for the system to be considered at rest
      */
      agx::Real getMaxRelativeSpeed() const;

      /**
      Assign maximum difference the 'speed'/desired speed parameter in a motor controller may change without
      splitting the constrained objects. Default: 1.0E-5.
      */
      void setMaxDesiredSpeedDiff( agx::Real value );

      /**
      \return the maximum difference the 'speed'/desired speed parameter in a motor controller may change without
              splitting the constrained objects
      */
      agx::Real getMaxDesiredSpeedDiff() const;

      /**
      Assign maximum difference the 'position'/desired angle parameter in a lock controller may change without
      splitting the constrained objects. Default: 1.0E-5.
      */
      void setMaxDesiredLockAngleDiff( agx::Real value );

      /**
      \return the maximum difference the 'position'/desired angle parameter in a lock controller may change without
              splitting the constrained objects
      */
      agx::Real getMaxDesiredLockAngleDiff() const;

      /**
      Assign maximum difference the 'position'/desired angle parameter in a range controller may change without
      splitting the constrained objects. Default: 1.0E-5.
      */
      void setMaxDesiredRangeAngleDiff( agx::Real value );

      /**
      \return the maximum difference the 'position'/desired angle parameter in a range controller may change without
      splitting the constrained objects
      */
      agx::Real getMaxDesiredRangeAngleDiff() const;

      /**
      Assign maximum difference the 'force range'/desired force range parameter in any controller may change without
      splitting the constrained objects. Default: 0.1.
      */
      void setMaxDesiredForceRangeDiff( agx::Real value );

      /**
      \return the maximum difference the 'force range'/desired force range parameter in any controller may change without
              splitting the constrained objects
      */
      agx::Real getMaxDesiredForceRangeDiff() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::ConstraintMergeSplitThresholds );

      /**
      Creates and returns a clone of this object.
      */
      virtual agx::ICloneableRef clone( agx::ICloneable* child = nullptr ) override;

      /**
      Resets all values to default.
      */
      virtual void resetToDefault() override;

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~ConstraintMergeSplitThresholds();
  };

  inline agx::Real ConstraintMergeSplitThresholds::getMaxRelativeSpeed() const
  {
    return get( MAX_RELATIVE_SPEED );
  }

  inline agx::Real ConstraintMergeSplitThresholds::getMaxDesiredSpeedDiff() const
  {
    return get( DESIRED_SPEED_DIFF );
  }

  inline agx::Real ConstraintMergeSplitThresholds::getMaxDesiredLockAngleDiff() const
  {
    return get( DESIRED_LOCK_ANGLE_DIFF );
  }

  inline agx::Real ConstraintMergeSplitThresholds::getMaxDesiredRangeAngleDiff() const
  {
    return get( DESIRED_RANGE_ANGLE_DIFF );
  }

  inline agx::Real ConstraintMergeSplitThresholds::getMaxDesiredForceRangeDiff() const
  {
    return get( DESIRED_FORCE_RANGE_DIFF );
  }
}
