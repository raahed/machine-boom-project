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

#include <agx/MergedBody.h>

namespace agxSDK
{
  class MergeSplitProperties;

  /**
  Object holding merged bodies and state given a pair of rigid bodies.
  */
  class AGXPHYSICS_EXPORT MergedState
  {
    public:
      enum EState
      {
        VALID                       = 1 << 0, /**< Valid state. */
        BOTH_ARE_NULL               = 1 << 1, /**< None of the bodies are merged. */
        ONE_IS_NULL                 = 1 << 2, /**< One of the bodies are merged. */
        BOTH_ARE_VALID              = 1 << 3, /**< Both of the bodies are merged. */
        MERGED_BODIES_ARE_SAME      = 1 << 4, /**< The bodies are merged together. */
        MERGED_BODIES_ARE_DIFFERENT = 1 << 5, /**< Both bodies are merged but has different parents (i.e., not merged with each other). */
        WIRE_BODY_INCLUDED          = 1 << 6,
        GRANULATE_BODY_INCLUDED     = 1 << 7
      };
      typedef agx::BitState< EState, agx::Int32 > State;

      enum ActionType : agx::Int16
      {
        MERGE        = 1 << 0,
        SPLIT_FIRST  = 1 << 1,
        SPLIT_SECOND = 1 << 2
      };
      typedef agx::BitState< ActionType, agx::Int16 > Actions;

    public:
      /**
      Creates an invalid merged state.
      */
      static MergedState invalid();

    public:
      /**
      Find merged state given two rigid bodies.
      \param rb1 - first rigid body
      \param rb2 - second rigid body
      */
      MergedState( agx::RigidBody* rb1, agx::RigidBody* rb2, const agxSDK::MergeSplitHandler* handler );

      /**
      Find merged state given two geometries.
      \param geometry1 - first geometry
      \param geometry2 - second geometry
      */
      MergedState( agxCollide::Geometry* geometry1, agxCollide::Geometry* geometry2, const agxSDK::MergeSplitHandler* handler );

      /**
      \return true if one or more masks matches current state
      */
      State getState() const;

      /**
      \return the allowed actions a merge split algorithm may perform
      */
      Actions getAllowedActions() const;

      /**
      \return the preferred merged body (larges if two, non-null if one)
      */
      agx::MergedBody* getMergedBody() const;

      /**
      \return the other (not preferred) if both bodies are merged, otherwise null
      */
      agx::MergedBody* getOtherMergedBody() const;

      /**
      \return the merged body associated to the first object during construct
      */
      agx::MergedBody* getMergedBody1() const;

      /**
      \return the merged body associated to the second object during construct
      */
      agx::MergedBody* getMergedBody2() const;

      /**
      \return first rigid body (coupled to getMergedBody1)
      */
      agx::RigidBody* getRigidBody1() const;

      /**
      \return second rigid body (coupled to getMergedBody2)
      */
      agx::RigidBody* getRigidBody2() const;

      /**
      \return properties used for first rigid body (the properties could belong to a geometry)
      */
      const agxSDK::MergeSplitProperties* getProperties1() const;

      /**
      \return properties used for second rigid body (the properties could belong to a geometry)
      */
      const agxSDK::MergeSplitProperties* getProperties2() const;

      /**
      \return true if first object during construct is merged
      */
      agx::Bool firstIsMerged() const;

      /**
      \return true if second object during construct is merged
      */
      agx::Bool secondIsMerged() const;

      /**
      \return true if the state is valid, i.e., a state different from NONE
      */
      agx::Bool isValid() const;

      /**
      Update the current state of how the bodies are merged and return this.
      */
      MergedState& update();

    protected:
      /**
      Delegating constructor.
      */
      void construct( agx::RigidBody* rb1, agx::RigidBody* rb2 );

    protected:
      agx::MergedBody*                    m_mergedBody[2];
      agx::MergedBody*                    m_preferredMergedBody;
      agx::RigidBody*                     m_rigidBodies[2];
      const agxSDK::MergeSplitProperties* m_properties[2];
      State                               m_state;
      Actions                             m_allowedActions;
  };

  class AGXPHYSICS_EXPORT ConstraintMergedState : public MergedState
  {
    public:
      /**
      Creates an invalid merged state.
      */
      static ConstraintMergedState invalid();

    public:
      /**
      Find merged state given constraint. If the constraint doesn't have properties,
      allowed actions will be NOTHING.
      \param constraint - constraint
      */
      ConstraintMergedState( const agx::HighLevelConstraintImplementation* constraint, const agxSDK::MergeSplitHandler* handler );

      /**
      \return the merge split properties of the constraint
      */
      const agxSDK::MergeSplitProperties* getConstraintProperties() const;

    protected:
      const agxSDK::MergeSplitProperties* m_constraintProperties;

    private:
      ConstraintMergedState();
  };

  AGX_FORCE_INLINE MergedState::State MergedState::getState() const
  {
    return m_state;
  }

  AGX_FORCE_INLINE MergedState::Actions MergedState::getAllowedActions() const
  {
    return m_allowedActions;
  }

  AGX_FORCE_INLINE agx::Bool MergedState::isValid() const
  {
    return m_state.Is( VALID );
  }
}
