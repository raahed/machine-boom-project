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

#include <agx/AddedMassInteraction.h>

namespace agx
{
  /**
  Class containing and handling a global view of several objects interacting
  trough strong added mass interactions.
  */
  class AGXPHYSICS_EXPORT AddedMassInteractionHandler : public agx::Interaction
  {
    public:
      /**
      Default constructor.
      */
      AddedMassInteractionHandler();

      /**
      Assign diagonal block \p matrix for \p rb. If the matrix is empty the interaction is removed.
      \param rb - rigid body associated to the matrix block located on the diagonal in the system matrix
      \param matrix - matrix data, given in \p rb center of mass coordinate frame, to be transformed and
                      written to the system matrix diagonal (if all zeros, nothing will be written)
      \return true if \p rb and \p block are accepted, otherwise false (rb == null or invalid matrix)
      */
      agx::Bool setBlock( agx::RigidBody* rb, const agx::AddedMassInteraction::Matrix6x6& matrix );

      /**
      Assign off diagonal matrix block \p matrix, defining an interaction between \p rb1 and \p rb2, i.e.,
      how acceleration of the first body results in a force on the second body. If the matrix is empty,
      the interaction is removed.
      \param rb1 - first rigid body (not valid if null), default defining in which coordinate frame \p matrix is given
      \param rb2 - second rigid body (not valid if null)
      \param matrix - matrix data, given in \p rb1 center of mass coordinate frame, to be transformed and
                      written to the system matrix off diagonal (if all zeros, nothing will be written)
      \return true if \p rb1, \p rb2 and \p matrix are accepted, otherwise false (rb1 == null, rb2 == null or invalid matrix)
      */
      agx::Bool setBlock( agx::RigidBody* rb1, agx::RigidBody* rb2, const agx::AddedMassInteraction::Matrix6x6& matrix );

      /**
      Removes garbage (interactions containing deleted and/or removed bodies) from the simulation.
      This method is automatically called one time each time step, before the solver runs.
      */
      void garbageCollect();

    protected:
      typedef agx::SymmetricPair< agx::RigidBody* > SymmetricRigidBodyPtrPair;
      typedef agx::HashVector< agx::RigidBody*, agx::AddedMassInteractionRef > RbAddedMassInteractionContainer;
      typedef agx::HashVector< SymmetricRigidBodyPtrPair, agx::AddedMassInteractionRef > RbRbAddedMassInteractionContainer;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~AddedMassInteractionHandler();

      /**
      Before solve, expand collected interactions for the solver.
      */
      virtual void preSystemCallback( agx::DynamicsSystem* dynamicsSystem ) override;

      /**
      After solve, remove interactions.
      */
      virtual void postSystemCallback( agx::DynamicsSystem* dynamicsSystem ) override;

    protected:
      /**
      Create key for two body interaction.
      */
      SymmetricRigidBodyPtrPair createKey( agx::RigidBody* rb1, agx::RigidBody* rb2 ) const;

      /**
      Creates an added mass interaction given a single body. Assumes rb != null.
      */
      agx::AddedMassInteraction* getOrCreateInteraction( agx::RigidBody* rb );

      /**
      Creates and added mass interaction given two bodies. Assumes rb1 and rb2 != null.
      */
      agx::AddedMassInteraction* getOrCreateInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2 );

      /**
      Removes a single body added mass interaction. Assumes rb1 != null.
      */
      void removeInteraction( agx::RigidBody* rb );

      /**
      Removes a two body added mass interaction. Assumes rb1 and rb2 != null.
      */
      void removeInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2 );

      /**
      Create a default added mass interaction given one or two rigid bodies.
      */
      agx::AddedMassInteractionRef createInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2 = nullptr ) const;

    private:
      RbAddedMassInteractionContainer            m_rbInteractions;
      RbRbAddedMassInteractionContainer          m_pairInteractions;
  };

  typedef ref_ptr< AddedMassInteractionHandler > AddedMassInteractionHandlerRef;
}

