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

#include <agxSDK/MergedBodySolverData.h>
#include <agxSDK/MergeSplitHandler.h>

#define REPORT_SYSTEM_JOB_ENABLE 1
#if REPORT_SYSTEM_JOB_ENABLE
#include <agxUtil/TimerBlock.h>
#include <agx/Thread.h>
#include <agx/macros.h>

// Since winbase.h defines an IGNORE macro, but we like to define an ENUM with a IGNORE value
#ifdef WIN32
#ifdef IGNORE
#undef IGNORE
#endif
#endif

#define REPORT_SYSTEM_JOB_EX( str )\
  auto sjrtb = agxUtil::TimerBlock( []( agx::UInt64 start, agx::UInt64 stop )\
  {\
    agx::Thread::getCurrentThread()->reportSystemJob( start, stop, str );\
  } )

#define REPORT_SYSTEM_JOB\
  const char* AGX_CONCAT(agx_functionName,__LINE__) = __FUNCTION__; \
  auto sjrtb = agxUtil::TimerBlock( [AGX_CONCAT(agx_functionName,__LINE__)]( agx::UInt64 start, agx::UInt64 stop )\
  {\
    agx::Thread::getCurrentThread()->reportSystemJob( start, stop, AGX_CONCAT(agx_functionName,__LINE__) );\
  } )
#else
#define REPORT_SYSTEM_JOB_EX( str )
#define REPORT_SYSTEM_JOB
#endif

#include <agxWire/Node.h>

namespace agxWire
{
  class WireDistanceCompositeConstraint;
}

namespace agxSDK
{
  namespace MergeSplitUtils
  {
    /**
    Calculates velocity at a given point given center of mass position, linear- and angular velocity.
    \param point - point in world
    \param cmPos - center of mass position
    \param linVel - linear velocity
    \param angVel - angular velocity
    \return velocity at given point
    */
    AGX_FORCE_INLINE agx::Vec3 calculateRelativePointVelocity( const agx::Vec3& point,
                                                               const agx::Vec3& cmPos,
                                                               const agx::Vec3& linVel,
                                                               const agx::Vec3& angVel )
    {
      return ( linVel + ( angVel ^ ( point - cmPos ) ) );
    }

    /**
    Calculates impact speed of one body given contact point and normal.
    \param point - contact point in world
    \param normal - contact normal in world
    \param cmPos - center of mass position of the body
    \param linVel - linear velocity of the body
    \param angVel - angular velocity of the body
    \return impact speed, > 0 if approaching
    */

    AGX_FORCE_INLINE agx::Real calculateImpactSpeed( const agx::Vec3& point,
                                                     const agx::Vec3f& normal,
                                                     const agx::Vec3& cmPos,
                                                     const agx::Vec3& linVel,
                                                     const agx::Vec3& angVel )
    {
      return -( agx::Vec3( normal ) * calculateRelativePointVelocity( point, cmPos, linVel, angVel ) );
    }

    /**
    Calculates relative velocity at a point relative to two bodies.
    \param point - point in world
    \param relVel - relative velocity, rb1.velocity - rb2.velocity
    \param cmPos1 - center of mass position of the body 1
    \param angVel1 - angular velocity of the body 1
    \param cmPos2 - center of mass position of the body 2
    \param angVel2 - angular velocity of the body 2
    \return the relative velcoty at the given point
    */
    AGX_FORCE_INLINE agx::Vec3 calculateRelativePointVelocity( const agx::Vec3& point,
                                                               const agx::Vec3& relVel,
                                                               const agx::Vec3& cmPos1,
                                                               const agx::Vec3& angVel1,
                                                               const agx::Vec3& cmPos2,
                                                               const agx::Vec3& angVel2 )
    {
      return ( relVel + ( angVel1 ^ ( point - cmPos1 ) ) + ( ( point - cmPos2 ) ^ angVel2 ) );
    }

    /**
    Calculates impact speed of two rigid bodies given contact point and normal.
    \param point - contact point in world
    \param normal - contact normal in world
    \param relVel - relative velocity, rb1.velocity - rb2.velocity
    \param cmPos1 - center of mass position of the body 1
    \param angVel1 - angular velocity of the body 1
    \param cmPos2 - center of mass position of the body 2
    \param angVel2 - angular velocity of the body 2
    \return impact speed, > 0 if approaching
    */
    AGX_FORCE_INLINE agx::Real calculateImpactSpeed( const agx::Vec3& point,
                                                     const agx::Vec3f& normal,
                                                     const agx::Vec3& relVel,
                                                     const agx::Vec3& cmPos1,
                                                     const agx::Vec3& angVel1,
                                                     const agx::Vec3& cmPos2,
                                                     const agx::Vec3& angVel2 )
    {
      return -( agx::Vec3( normal ) * calculateRelativePointVelocity( point, relVel, cmPos1, angVel1, cmPos2, angVel2 ) );
    }

    /**
    Calculates force frame (edge strength) given reference rigid body, contact edge and optionally
    an merge split handler (to use contact thresholds).
    \param refRb - reference rigid body
    \param contactEdge - geometry contact edge
    \param handler - if given, any suitable contact threshold will be used, otherwise default
    \return edge strength force frame
    */
    AGXPHYSICS_EXPORT agxSDK::MergedBodySolverData::ForceFrame calculateContactForceFrame( const agx::RigidBody* refRb,
                                                                                           const agx::MergedBody::GeometryContactEdgeInteraction* contactEdge,
                                                                                           const MergeSplitHandler* handler = nullptr );

    struct ExternalForceSplitResult
    {
      ExternalForceSplitResult()
        : shouldSplit( false ), rbTop( nullptr ), rbBottom( nullptr ) {}
      agx::Bool shouldSplit;
      const agx::RigidBody* rbTop;
      const agx::RigidBody* rbBottom;
    };

    /**
    Checks if two bodies should split given edge strength force frame and external forces.
    \param edgeStrength - force frame describing the strength of the connection between \p rb and \p otherRb
    \param rb - first rigid body
    \param otherRb - other rigid body
    \param externalForces - external forces (which could result in the split) - any gravity force should be included here
    \param externalTorques - external torques (not used)
    \param gravity - gravity acceleration used to estimate rbTop and rbBottom
    \param handler - merge split handler with thresholds
    \return split result
    */
    AGXPHYSICS_EXPORT ExternalForceSplitResult shouldSplitGivenExternalForce( agxSDK::MergedBodySolverData::ForceFrame edgeStrength,
                                                                              const agx::RigidBody* rb,
                                                                              const agx::RigidBody* otherRb,
                                                                              const agx::Vec3& externalForces,
                                                                              const agx::Vec3& externalTorques,
                                                                              const agx::Vec3& gravity,
                                                                              const agxSDK::MergeSplitHandler& handler );

    template<typename T>
    using CollectConnectingEdgesPtrContainer = agx::Vector<T*>;

    /**
    Collects edges of given type/tag that are connected within a merged body.
    \param referenceEdge - reference edge (starting point is referenceEdge.getRigidBody1())
    \param edgeTag - edge tag matching T
    \param bounceAtNonSplittableBodies - stop search at bodies that can't split? Example: 10 cranes on
                                         a non-splittable ship parent where referenceEdge is a prismatic
                                         in one of the cranes. With bounceAtNonSplittableBodies = true,
                                         the result will only contain edges from the crane with the prismatic,
                                         but with bounceAtNonSplittableBodies = false all 10 cranes will be
                                         included.
    \param mergedBody - merged body with the referenceEdge
    \param handler - the merge split handler
    \param[out] result - result with connected edges of the given type/tag
    */
    template<typename T>
    void collectConnectingEdges( const T& referenceEdge,
                                 agx::MergedBody::EdgeInteraction::InteractionTag edgeTag,
                                 agx::Bool bounceAtNonSplittableBodies,
                                 const agx::MergedBody& mergedBody,
                                 const MergeSplitHandler& handler,
                                 CollectConnectingEdgesPtrContainer<T>& result )
    {
      using VisitedContainer          = agx::HashSet<const agx::RigidBody*>;
      using SymmetricRigidBodyPtrPair = agx::SymmetricPair<const agx::RigidBody*>;
      using VisitedPairContainer      = agx::HashSet<SymmetricRigidBodyPtrPair>;
      using BfsQueueContainer         = std::queue<const agx::RigidBody*>;

      VisitedContainer visited;
      VisitedPairContainer visitedPairs;
      BfsQueueContainer bfsQueue;

      const auto visit = [&visited, &bfsQueue]( const agx::RigidBody* rb )
      {
        agxAssert( !visited.contains( rb ) );
        visited.insert( rb );
        bfsQueue.push( rb );
      };

      const auto visitedPair = [&visitedPairs]( const agx::RigidBody* rb1, const agx::RigidBody* rb2 ) -> agx::Bool
      {
        SymmetricRigidBodyPtrPair rb1Rb2Pair( rb1, rb2 );
        if ( visitedPairs.contains( rb1Rb2Pair ) )
          return true;

        visitedPairs.insert( rb1Rb2Pair );

        return false;
      };

      const auto pop = [&bfsQueue]() -> const agx::RigidBody*
      {
        agxAssert( !bfsQueue.empty() );
        auto front = bfsQueue.front();
        bfsQueue.pop();
        return front;
      };

      visit( referenceEdge.getRigidBody1() );
      while ( !bfsQueue.empty() ) {
        const auto rb        = pop();
        const auto neighbors = mergedBody.getNeighbors( rb );

        agxAssert( neighbors != nullptr );
        for ( const auto neighbor : *neighbors ) {
          if ( visitedPair( neighbor, rb ) )
            continue;

          const auto edges = mergedBody.getEdges( neighbor, rb );
          agxAssert( edges != nullptr );
          auto oneHasEdgeTag = false;
          for ( const auto& edge : *edges ) {
            const auto edgeHasTag = edge->isTagged( edgeTag );
            oneHasEdgeTag         = oneHasEdgeTag || edgeHasTag;
            if ( edge != &referenceEdge && edgeHasTag ) {
              agxAssert( !result.contains( edge ) );
              result.push_back( edge->template as<T>() );
            }
          }

          // Continue search if rb<->neighbor has the given tag and hasn't already
          // been visited. Stop if:
          //   bounceAtNonSplittableBodies && !handler.maySplit( neighbor )
          const auto visitThisNeighbor = neighbor != nullptr &&
                                         oneHasEdgeTag &&
                                         ( !bounceAtNonSplittableBodies || handler.maySplit( neighbor ) ) &&
                                         !visited.contains( neighbor );
          if ( visitThisNeighbor )
            visit( neighbor );
        }
      }
    }

    class WireSegmentMergedState;
    using WireSegmentMergedStateContainer = agx::VectorPOD<WireSegmentMergedState>;

    /**
    Wire merged state segment along a wire. The state of this segment can be:
      FREE: None of the nodes in this segment is merged.
      MERGED: All nodes in this segment is merged to the same merged body instance.
      IGNORE: All nodes in this segment is explicitly merged.

    The state is only checked for lumped nodes in the wire. Intermediate contact
    and/or eye nodes are included in the segment but the merged state may differ.

    Note: Say that the wire has 10 nodes - all are merged. Nodes 1 to 6 are merged
          to mergedBody1 and 7 to 10 are merged to mergedBody2. Calling:
            auto segments = WireMergedState::create( wire );
          will contain two entries, both with state MERGED but since the MergedBody
          instances differs, the segments are separated.
    */
    class AGXPHYSICS_EXPORT WireSegmentMergedState
    {
      public:
        /**
        Creates vector of WireMergedState segments along the wire starting from
        first lumped node to (and including) last lumped node. Note that the
        first and last attachment nodes are both excluded.
        \param wire - wire instance
        \return vector of merged states along the wire
        */
        static WireSegmentMergedStateContainer create( const agxWire::Wire* wire );

      public:
        enum EState : agx::UInt32
        {
          FREE   = 1 << 0, /**< Segment isn't merged. */
          MERGED = 1 << 1, /**< Segment is merged by the MergeSplitHandler. */
          IGNORE = 1 << 2  /**< Segemnt is explicitly merged by the user. */
        };
        using State = agx::BitState<EState, agx::UInt32>;

      public:
        /**
        Construct given wire constraint.
        \param wire - wire constraint
        */
        WireSegmentMergedState( const agxWire::WireDistanceCompositeConstraint* wire );

        /**
        Initialize using iterator to first node (end of previous segment).
        \param it - first node iterator in this segment
        \return end iterator of this segment
        */
        agxWire::NodeConstIterator construct( agxWire::NodeConstIterator it );

        /**
        \return the state of this segment
        */
        State getState() const;

        /**
        \return the number of nodes in this segment
        */
        agx::UInt size() const;

        /**
        \return begin node iterator of this segment
        */
        agxWire::NodeConstIterator begin() const;

        /**
        \return end node iterator of this segment
        */
        agxWire::NodeConstIterator end() const;

        /**
        \return the first node in this segment (by definition a lumped node)
        */
        agxWire::BodyFixedNode* front() const;

        /**
        \return the last node in this segment
        */
        agxWire::Node* back() const;

        /**
        \return merged body instance if state == MERGED - otherwise nullptr
        */
        agx::MergedBody* getMergedBody() const;

      private:
        const agxWire::WireDistanceCompositeConstraint* m_wire;
        agx::MergedBody* m_mergedBody;
        State m_state;
        agxWire::NodeConstIterator m_begin;
        agxWire::NodeConstIterator m_end;
        agx::UInt m_size;
    };
  }
}
