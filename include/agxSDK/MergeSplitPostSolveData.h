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

#include <agxSDK/MergedState.h>

#include <agx/Jacobian.h>

#include <agx/Physics/BinaryConstraintEntity.h>
#include <agx/Physics/ManyBodyConstraintEntity.h>
#include <agx/Physics/GeometryContactEntity.h>
#include <agx/Physics/ContactConstraintEntity.h>
#include <agx/Physics/ConstraintRowEntity.h>
#include <agx/Physics/ParticlePairContactEntity.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>

#include <agx/Physics/Partitioner/GraphNode.h>

namespace agxSDK
{
  class MergeSplitHandler;

  struct AGXPHYSICS_EXPORT MergeSplitPostSolveData
  {
    MergeSplitPostSolveData( agx::Physics::GraphNodeData& graphNode,
                             agx::Physics::BinaryConstraintData& binaryConstraint,
                             agx::Physics::ConstraintRowData& binaryConstraintRow,
                             const agxData::Array< agx::Jacobian6DOFElement >& binaryConstraintJacobian,
                             agx::Physics::ManyBodyConstraintData& manyBodyConstraint,
                             agx::Physics::ConstraintRowData& manyBodyConstraintRow,
                             const agxData::Array< agx::Jacobian6DOFElement >& manyBodyConstraintJacobian,
                             agx::Physics::ContactConstraintData& contactConstraint,
                             agx::Physics::ConstraintRowData& contactConstraintRow,
                             const agxData::Array< agx::Jacobian6DOFElement >& contactConstraintJacobian,
                             agx::Physics::GeometryContactData& geometryContact,
                             agx::Physics::RigidBodyData& rigidBody );

    /**
    \return the graph node type at given index
    */
    agx::GraphNode::Type getType( agx::UInt index ) const;

    /**
    \return edge as binary constraint (check validity)
    */
    agx::Physics::BinaryConstraintPtr asBinaryConstraint( agx::UInt index ) const;

    /**
    \return edge as many body constraint (check validity)
    */
    agx::Physics::ManyBodyConstraintPtr asManyBodyConstraint( agx::UInt index ) const;

    /**
    \return edge as contact constraint (check validity)
    */
    agx::Physics::ContactConstraintPtr asContactConstraint( agx::UInt index ) const;

    /**
    \return edge as geometry contact (check validity)
    */
    agx::Physics::GeometryContactPtr asGeometryContact( agx::UInt index ) const;

    /**
    \return edge as particle-particle contact (check validity)
    */
    agx::Physics::ParticlePairContactPtr asParticlePairContactPtr( agx::UInt index ) const;

    /**
    \return edge as particle-geometry contact (check validity)
    */
    agx::Physics::ParticleGeometryContactPtr asParticleGeometryContactPtr( agx::UInt index ) const;

    /**
    \return edge as rigid body (check validity)
    */
    agx::Physics::RigidBodyPtr asRigidBody( agx::UInt index ) const;

    /**
    \note If the edge isn't a pair (e.g., many body constraint or body) the state is invalid
    \return a merged pair state given the current edge
    */
    MergedState createMergedState( agx::UInt index, const agxSDK::MergeSplitHandler& handler ) const;

    agx::Physics::GraphNodeData&                      graphNode;
    agx::Physics::BinaryConstraintData&               binaryConstraint;
    agx::Physics::ConstraintRowData&                  binaryConstraintRow;
    const agxData::Array< agx::Jacobian6DOFElement >& binaryConstraintJacobian;
    agx::Physics::ManyBodyConstraintData&             manyBodyConstraint;
    agx::Physics::ConstraintRowData&                  manyBodyConstraintRow;
    const agxData::Array< agx::Jacobian6DOFElement >& manyBodyConstraintJacobian;
    agx::Physics::ContactConstraintData&              contactConstraint;
    agx::Physics::ConstraintRowData&                  contactConstraintRow;
    const agxData::Array< agx::Jacobian6DOFElement >& contactConstraintJacobian;
    agx::Physics::GeometryContactData&                geometryContact;
    agx::Physics::RigidBodyData&                      rigidBody;

    MergeSplitPostSolveData& operator = ( const MergeSplitPostSolveData& ) = delete;
  };

  inline agx::GraphNode::Type MergeSplitPostSolveData::getType( agx::UInt index ) const
  {
    return (agx::GraphNode::Type)graphNode.type[ index ];
  }

  inline agx::Physics::BinaryConstraintPtr MergeSplitPostSolveData::asBinaryConstraint( agx::UInt index ) const
  {
    return graphNode.source[ index ];
  }

  inline agx::Physics::ManyBodyConstraintPtr MergeSplitPostSolveData::asManyBodyConstraint( agx::UInt index ) const
  {
    return graphNode.source[ index ];
  }

  inline agx::Physics::ContactConstraintPtr MergeSplitPostSolveData::asContactConstraint( agx::UInt index ) const
  {
    return contactConstraint.instance[ asGeometryContact( index ).constraintIndex() ];
  }

  inline agx::Physics::GeometryContactPtr MergeSplitPostSolveData::asGeometryContact( agx::UInt index ) const
  {
    return graphNode.source[ index ];
  }

  inline agx::Physics::ParticlePairContactPtr MergeSplitPostSolveData::asParticlePairContactPtr( agx::UInt index ) const
  {
    return graphNode.source[ index ];
  }

  inline agx::Physics::ParticleGeometryContactPtr MergeSplitPostSolveData::asParticleGeometryContactPtr( agx::UInt index ) const
  {
    return graphNode.source[ index ];
  }

  inline agx::Physics::RigidBodyPtr MergeSplitPostSolveData::asRigidBody( agx::UInt index ) const
  {
    return graphNode.source[ index ];
  }
}
