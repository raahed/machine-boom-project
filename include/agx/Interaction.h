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

#include <agxSDK/SimulationProxy.h>

namespace agx
{
  class DynamicsSystem;
  AGX_DECLARE_POINTER_TYPES( Interaction );
  typedef agx::SetVector<agx::InteractionRef> InteractionRefSetVector;


  /**
  The base class for interactions.
  */
  class AGXPHYSICS_EXPORT Interaction : public Referenced, public virtual agxStream::Serializable
  {
    public:
      /**
      Default constructor.
      */
      Interaction();

      /**
      Name this interaction.
      \param name - new name of this interaction
      */
      void setName( const agx::Name& name );

      /**
      \return the name of this interaction
      */
      const agx::Name& getName() const;

      /**
      Assign a new property container.
      \param propertyContainer -  new property container
      */
      void setPropertyContainer( agx::PropertyContainer* propertyContainer );

      /**
      \return a property container (if it doesn't exist, a new will be created)
      */
      agx::PropertyContainer* getPropertyContainer();

      /**
      \return a property container (if it doesn't exist, a new will be created)
      */
      const agx::PropertyContainer* getPropertyContainer() const;

      /**
      \return true if a property container has been created, otherwise false
      */
      agx::Bool hasPropertyContainer() const;

      /**
      \return the simulation proxy for this strong interaction
      */
      agxSDK::SimulationProxy* getSimulationProxy() const;

    public:
      /**
      This virtual method is implemented so that it computes the
      interaction forces on all the physical bodies implicated.
      This method defaults to doing nothing.  Constraints and other
      strong interactions might not implement anything for this because the
      forces they generate are computed by a solver.  A weak interaction
      must implement this method to have any effect.
      \param dynamicsSystem - the system to which the forces should be added
      */
      virtual void updateForce(agx::DynamicsSystem* dynamicsSystem);

#ifndef SWIG
      /**
      Called after all pre step events, before solve.
      */
      virtual void preSystemCallback( agx::DynamicsSystem* /*dynamicsSystem*/ ) {}
      /**
      Called after all post step events, after solve.
      */
      virtual void postSystemCallback( agx::DynamicsSystem* /*dynamicsSystem*/ ) {}
#endif

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Interaction );

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~Interaction() {}

      /**
      Called when added to a simulation.
      */
      virtual void addNotification();

      /**
      Called when removed from a simulation.
      */
      virtual void removeNotification();

      friend class agxSDK::Simulation;
      /**
      Assign simulation. Be careful.
      */
      virtual void setSimulation( agxSDK::Simulation* simulation );

    private:
      mutable PropertyContainerRef m_propertyContainer;
      Name m_name;
      agxSDK::SimulationProxyRef m_simulationProxy;
  };

  inline void Interaction::updateForce(agx::DynamicsSystem* /*dynamicsSystem*/) {}

}
