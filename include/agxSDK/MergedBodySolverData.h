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

#include <agxSDK/MergeSplitPostSolveData.h>

#include <agx/MergedBody.h>

namespace agxSDK
{
  class GeometryContactMergeSplitAlgorithm;
  class ConstraintMergeSplitAlgorithm;

  /**
  Collection of external (active) interactions, interacting with merged rigid bodies.
  */
  class AGXPHYSICS_EXPORT MergedBodySolverData
  {
    public:
      /**
      Force frame where the resultant force is u + v + n.
      */
      struct ForceFrame
      {
        agx::Vec3 u; /**< Force in u direction. */
        agx::Vec3 v; /**< Force in v direction. */
        agx::Vec3 n; /**< Force in n direction. */
      };

      /**
      Object containing force and torque from a graph node.
      */
      struct ForceData
      {
        ForceData()
          : graphNodeIndex( agx::InvalidIndex )
        {
        }

        ForceData( agx::UInt graphNodeIndex,
                   const agx::Vec3& force,
                   const agx::Vec3& torque )
          : graphNodeIndex( graphNodeIndex ), force( force ), torque( torque )
        {
        }

        agx::UInt graphNodeIndex; /**< Graph node index such that type = postSolveData.getType( graphNodeIndex ). */
        agx::Vec3 force;          /**< Force in world coordinates. */
        agx::Vec3 torque;         /**< Torque in world coordinates. */
      };

    public:
      using ForceDataVector                   = agx::VectorPOD<ForceData>;
      using BodyForceDataContainer            = agx::HashVector<const agx::RigidBody*, ForceDataVector>;
      using GlobalGraphNodeIndexContainer     = agx::UIntVector;
      using MergedBodyGraphNodeIndexContainer = agx::HashTable<const agx::MergedBody*, GlobalGraphNodeIndexContainer>;

    public:
      /**
      Default constructor.
      */
      MergedBodySolverData();

      /**
      Collect data given sources.
      */
      void collectData( const agx::MergedBody* mergedBody,
                        const MergeSplitPostSolveData& data,
                        const GlobalGraphNodeIndexContainer* externalInteractions,
                        const GeometryContactMergeSplitAlgorithm* geometryContactAlgorithm,
                        const ConstraintMergeSplitAlgorithm* constraintAlgorithm );

      /**
      \return force data container {rb: [ForceData]}
      */
      const BodyForceDataContainer& getForceData() const;

      /**
      \return force data for given rigid body if present - otherwise null
      */
      const ForceDataVector* tryGetForceData( const agx::RigidBody* rb ) const;

      /**
      \return the merged body
      */
      const agx::MergedBody* getMergedBody() const;

      /**
      Calculates the total contact edge strength given merged rigid body.
      \param rb - merged rigid body such that this->getMergedBody() == agx::MergedBody::getActive( rb )
      \param[out] contactStrength - contact force frame reprecenting how hard \p rb is merged to this merged body
      \return true if data was added to \p contactStrength
      */
      agx::Bool calculateContactEdgeStrength( const agx::RigidBody* rb,
                                              MergedBodySolverData::ForceFrame& contactStrength ) const;

      /**
      Visitor callback where ForceFrame is edge strength in world coordinates and 'otherRb'.
      */
      using ContactEdgeStrengthVisitor = std::function<void( ForceFrame, const agx::RigidBody* )>;

      /**
      Predicate to calculate force frame and visit ContactEdgeStrengthVisitor.
      */
      using ContactEdgeStrengthVisitorPredicate = std::function<agx::Bool( const agx::RigidBody* )>;

      /**
      Traverses all contact edges associated to \p rb where the visitor arguments are the edge strength
      and other rigid body in the interaction.
      \param rb - rigid body to traverse contact edges for
      \param visitor - edge strength and other rigid body visitor
      */
      void traverseContactEdgeStrength( const agx::RigidBody* rb,
                                        ContactEdgeStrengthVisitorPredicate pred,
                                        ContactEdgeStrengthVisitor visitor ) const;

    private:
      const agx::MergedBody* m_mergedBody;
      GlobalGraphNodeIndexContainer m_nodes;
      BodyForceDataContainer m_bodyForceDataContainer;
  };

  inline const agx::MergedBody* MergedBodySolverData::getMergedBody() const
  {
    return m_mergedBody;
  }
}
