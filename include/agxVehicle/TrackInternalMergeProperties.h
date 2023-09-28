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

#include <agxVehicle/export.h>

#include <agx/BitState.h>

namespace agxVehicle
{
  /**
  Properties and thresholds for internal merging of nodes in agxVehicle::Track.
  */
  class AGXVEHICLE_EXPORT TrackInternalMergeProperties
  {
    public:
      /**
      Contact reduction of merged nodes in contact with other objects
      such as ground.
      */
      enum ContactReduction
      {
        NONE,      /**< Contact reduction disabled. */
        MINIMAL,   /**< Contact reduction enabled with bin resolution = 3. */
        MODERATE,  /**< Contact reduction enabled with bin resolution = 2. */
        AGGRESSIVE /**< Contact reduction enabled with bin resolution = 1. */
      };

    public:
      /**
      Default constructor:
        merge enabled                     = false
        number of nodes per merge segment = 3
        contact reduction                 = ContactReduction::MINIMAL
        lock to reach merge condition     = true
        lock compliance                   = 1.0E-11
        lock damping                      = 3 / 60
        max angle merge condition         = 1.0E-5 (radians)
      */
      TrackInternalMergeProperties();

      /**
      Enable/disable merge of nodes to segments in the track.
      \param enable - true to enable, false to disable
      */
      void setEnableMerge( agx::Bool enable );

      /**
      \return true if merge functionality is enabled, otherwise false
      */
      agx::Bool getEnableMerge() const;

      /**
      Assign number of nodes in a row that may merge together.
      \param numNodesPerMergeSegment - number of nodes in a row that may merge together
      */
      void setNumNodesPerMergeSegment( agx::UInt numNodesPerMergeSegment );

      /**
      \return number of nodes in a row that may merge together
      */
      agx::UInt getNumNodesPerMergeSegment() const;

      /**
      Set contact reduction level of merged nodes against other objects.
      \param contactReductionLevel - contact reduction level
      */
      void setContactReduction( ContactReduction contactReductionLevel );

      /**
      \return contact reduction level
      */
      ContactReduction getContactReduction() const;

      /**
      \return true if contact reduction is enabled (i.e., MINIMAL, MODERATE or AGGRESSIVE)
      */
      agx::Bool getEnableContactReduction() const;

      /**
      \return bin resolution of contact reduction - agx::InvalidIndex if contact reduction is disabled
      */
      agx::UInt getContactReductionBinResolution() const;

      /**
      Enable/disable the usage of hinge lock to reach merge condition (angle close to zero).
      \param enable - true to enable, false to disable
      */
      void setEnableLockToReachMergeCondition( agx::Bool enable );

      /**
      \return true if lock to reach merge condition is enabled, otherwise false
      */
      agx::Bool getEnableLockToReachMergeCondition() const;

      /**
      Assign compliance of the hinge lock used to reach merge condition.
      \param compliance - hinge lock compliance
      */
      void setLockToReachMergeConditionCompliance( agx::Real compliance );

      /**
      \return hinge lock compliance
      */
      agx::Real getLockToReachMergeConditionCompliance() const;

      /**
      Assign damping of the hinge lock used to reach merge condition.
      \param damping - hinge lock damping
      */
      void setLockToReachMergeConditionDamping( agx::Real damping );

      /**
      \return hinge lock damping
      */
      agx::Real getLockToReachMergeConditionDamping() const;

      /**
      Assign maximum angle > 0 to trigger merge between nodes. I.e., when
      the angle between two nodes < maxAngleToMerge the nodes will merge.
      \param maxAngleToMerge - maximum angle > 0 to trigger merge between two nodes
      */
      void setMaxAngleMergeCondition( agx::Real maxAngleToMerge );

      /**
      \return maximum angle > 0 that triggers merge between two nodes
      */
      agx::Real getMaxAngleMergeCondition() const;

      /**
      Resets all values to default (see default constructor).
      */
      void resetToDefault();

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Stores data to stream.
      */
      void store( agxStream::OutputArchive& out ) const;

      /**
      Restores data from stream.
      */
      void restore( agxStream::InputArchive& in );

      /**
      Data changed should induce complete split of all segments. E.g.,
      number of nodes per segment, disable of merge. The dirty flag
      will be set to false after calling this method.
      */
      agx::Bool getAndResetSplitRelatedDataDirtyFlag();
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      enum StateEnum : agx::UInt32
      {
        MERGE_ENABLE                       = 1 << 0,
        CONTACT_REDUCTION_LEVEL_AGGRESSIVE = 1 << 1,
        CONTACT_REDUCTION_LEVEL_MODERATE   = 1 << 2,
        CONTACT_REDICTION_LEVEL_MINIMAL    = 1 << 3,
        LOCK_TO_REACH_MERGE_CONDITION      = 1 << 4,
        SPLIT_RELATED_DATA_DIRTY           = 1 << 5
      };
      using State = agx::BitState<StateEnum, agx::UInt32>;

    private:
      State m_state;
      agx::UInt m_numNodesPerMergeSegment;
      agx::Real m_lockToReachMergeConditionCompliance;
      agx::Real m_lockToReachMergeConditionDamping;
      agx::Real m_maxAngleMergeCondition;
  };
}
