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

#include <agxSDK/Simulation.h>
#include <agx/StrongInteraction.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxWire
{
  /**
  Class to 'hide' getSimulation method in parent class and adding
  simulation proxy interface.
  */
  template< typename T >
  class SimulationProxyT : public T
  {
    public:
      agxSDK::SimulationProxy* getSimulationProxy() const { return m_simulationProxy; }

    protected:
      SimulationProxyT()
       : m_simulationProxy( agxSDK::Simulation::createDefaultProxy( nullptr ) ) {}

      virtual ~SimulationProxyT() {}

      /**
      Add notification if agxSDK::Assembly.
      */
      virtual void addNotification( agxSDK::Simulation* simulation ) { handleAddNotification( simulation ); }

      /**
      Remove notification if agxSDK::Assembly.
      */
      virtual void removeNotification( agxSDK::Simulation* simulation ) { handleRemoveNotification( simulation ); }

      /**
      Add notification if event type.
      */
      virtual void addNotification() { handleAddNotification( getSimulation() ); }

      /**
      Remove notification if event type.
      */
      virtual void removeNotification() { handleRemoveNotification( getSimulation() ); }

      /**
      Call this method when the simulation proxy should transform from local
      to global (i.e., the one owned by the simulation).
      */
      virtual void handleAddNotification( agxSDK::Simulation* simulation )
      {
        if ( simulation != nullptr ) {
          simulation->commit( m_simulationProxy );
          m_simulationProxy = simulation->getSimulationProxy();
        }
      }

      /**
      Call this method when the simulation proxy should transform form global
      back to local.
      */
      virtual void handleRemoveNotification( agxSDK::Simulation* /*simulation*/ )
      {
        agxAssert( m_simulationProxy->getSimulation() == nullptr || m_simulationProxy->empty() );
        m_simulationProxy = agxSDK::Simulation::createDefaultProxy( nullptr );
      }

    protected:
      virtual void addNotification(agxSDK::Assembly*) {};
      virtual void removeNotification(agxSDK::Assembly*) {};

    private:
      agxSDK::Simulation* getSimulation() { return T::getSimulation(); }
      const agxSDK::Simulation* getSimulation() const { return T::getSimulation(); }



    private:
      agxSDK::SimulationProxyRef m_simulationProxy;
  };

  /**
  Interface for simulation used by wire callbacks running in parallel.
  */
  class AGXPHYSICS_EXPORT SimulationProxy : public agxSDK::SimulationProxy
  {
    public:
      /**
      Construct given simulation. This object isn't valid without a simulation.
      \param simulation - simulation this object is proxy for
      */
      SimulationProxy( agxSDK::Simulation* simulation );

      /**
      Proxy method, add geometry. The geometry will be added to space as soon
      as state is serial.
      */
      virtual bool add( agxCollide::Geometry* geometry ) override;

      /**
      Proxy method, remove geometry. The geometry will be removed from space as soon
      as state is serial.
      */
      virtual bool remove( agxCollide::Geometry* geometry ) override;

      /**
      Proxy method, enable geometry. The geometry will be enabled/disabled as soon
      as state is serial.
      */
      virtual void setEnable( agxCollide::Geometry* geometry, bool enable ) override;

      /**
      Proxy method, add group ID to geometry. The ID will be added from the geometry as soon
      as state is serial.
      */
      virtual void addGroup( agxCollide::Geometry* geometry, agx::UInt32 id ) override;

      /**
      Proxy method, add group name to geometry. The group name will be added from the geometry as soon
      as state is serial.
      */
      virtual void addGroup( agxCollide::Geometry* geometry, const agx::Name& name ) override;

      /**
      Proxy method, remove group ID to geometry. The ID will be removed from the geometry as soon
      as state is serial.
      */
      virtual void removeGroup( agxCollide::Geometry* geometry, agx::UInt32 id ) override;

      /**
      Proxy method, remove group name to geometry. The group name will be removed from the geometry as soon
      as state is serial.
      */
      virtual void removeGroup( agxCollide::Geometry* geometry, const agx::Name& name ) override;

      /**
      Proxy method, add rigid body. The rigid body will be added to simulation as soon
      as state is serial.
      */
      virtual bool add( agx::RigidBody* body, bool addGeometries = true ) override;

      /**
      Proxy method, remove rigid body. The rigid body will be removed from the simulation as soon
      as state is serial.
      */
      virtual bool remove( agx::RigidBody* body, bool removeGeometries = true ) override;

      /**
      Proxy method, enable rigid body. The rigid body will be enabled/disabled as soon
      as state is serial.
      */
      virtual void setEnable( agx::RigidBody* body, bool enable ) override;

      /**
      Proxy method, add constraint. The constraint will be added to simulation as soon
      as state is serial.
      */
      virtual bool add( agx::Constraint* constraint ) override;

      /**
      Proxy method, remove constraint. The constraint will be removed from the simulation as soon
      as state is serial.
      */
      virtual bool remove( agx::Constraint* constraint ) override;

      /**
      Proxy method, add strong interaction. The strong interaction will be added to simulation as soon
      as state is serial.
      */
      virtual bool add( agx::StrongInteraction* si ) override;

      /**
      Proxy method, remove strong interaction. The strong interaction will be removed from the simulation as soon
      as state is serial.
      */
      virtual bool remove( agx::StrongInteraction* si ) override;

      /**
      Proxy method, add assembly. The assembly will be added to simulation as soon
      as state is serial.
      */
      virtual bool add( agxSDK::Assembly* assembly ) override;

      /**
      Proxy method, remove assembly. The assembly will be removed from the simulation as soon
      as state is serial.
      */
      virtual bool remove( agxSDK::Assembly* assembly ) override;

      /**
      Proxy method, add event listener. The event listener will be added to simulation as soon
      as state is serial.
      */
      virtual bool add( agxSDK::EventListener* listener ) override;

      /**
      Proxy method, remove event listener. The event listener will be removed from the simulation as soon
      as state is serial.
      */
      virtual bool remove( agxSDK::EventListener* listener ) override;

      /**
      Proxy method, agxCollide::Space::addGeometryContacts. The geometry contacts will be added
      as soon as state is serial.
      */
      virtual void addGeometryContacts( const agxCollide::LocalGeometryContactVector& contacts ) override;

      /**
      Proxy method, agx::ContactMaterial::transfer. The material will be transfered to the
      default storage as soon as state is serial.
      */
      virtual void transferToDefaultStorage( agx::ContactMaterial* material ) override;

      /**
      Proxy method, agx::ContactMaterial::transfer. The material will be transfered to the
      simulation/material manager storage as soon as state is serial.
      */
      virtual void transferToSimulationStorage( agx::ContactMaterial* material ) override;

      /**
      \return true if this proxy is empty
      */
      virtual bool empty() const override;

      /**
      \return true if state is parallel - otherwise false
      */
      virtual bool getRunningParallel() const override;

      /**
      Prepare method for parallel run.
      */
      virtual void preParallelRun() override;

      /**
      Post-parallel, all states/actions will be executed (e.g., add rigid bodies to simulation etc).
      */
      virtual void postParallelRun() override;

      /**
      Commit current content to a simulation. If the simulation
      given during construct or via setSimulation is non-zero:
      m_simulation == simulation.
      */
      virtual void commit( agxSDK::Simulation* simulation ) override;

      /**
      \return actual or guessed time step
      */
      virtual agx::Real getTimeStep() const override;

      /**
      \return gravity field if present - otherwise 0
      */
      virtual const agx::GravityField* getGravityField() const override;

      /**
      \return material manager if present - otherwise 0
      */
      virtual const agxSDK::MaterialManager* getMaterialManager() const override;

      /**
      \return the simulation
      */
      virtual const agxSDK::Simulation* getSimulation() const override;

      /**
      Assign new simulation.
      */
      virtual void setSimulation( agxSDK::Simulation* simulation ) override;

      /**
      Non-const simulation. Use with extreme care during parallel run!
      */
      virtual agxSDK::Simulation* getSimulationUnsafe() const override;

    protected:
      /**
      Protected destructor, reference counted object.
      */
      virtual ~SimulationProxy() {}

      /**
      \return true if the current mode is 'store local' (i.e., not send to simulation)
      */
      agx::Bool storeLocal() const;

    protected:
      /**
      Internal class to handle add/remove of "general" object.
      */
      template< typename T >
      class AddRemoveCompatibleObject
      {
        public:
          AddRemoveCompatibleObject() {}

          bool add( T* object )
          {
            if ( object == nullptr || m_objectsToAdd.contains( object ) )
              return false;

            agx::ScopeLock< agx::Mutex > sl( m_mutex );

            m_objectsToAdd.push_back( object );
            m_objectsToRemove.findAndErase( object );

            return true;
          }

          bool remove( T* object )
          {
            if ( object == nullptr || m_objectsToRemove.contains( object ) )
              return false;

            agx::ScopeLock< agx::Mutex > sl( m_mutex );

            m_objectsToRemove.push_back( object );
            m_objectsToAdd.findAndErase( object );

            return true;
          }

          void commit( agxSDK::Simulation* simulation )
          {
            while ( !m_objectsToAdd.empty() ) {
              simulation->add( m_objectsToAdd.back() );
              m_objectsToAdd.pop_back();
            }

            while ( !m_objectsToRemove.empty() ) {
              simulation->remove( m_objectsToRemove.back() );
              m_objectsToRemove.pop_back();
            }
          }

          bool empty() const { return m_objectsToAdd.empty() && m_objectsToRemove.empty(); }

        protected:
          typedef agx::Vector< agx::ref_ptr< T > > ObjRefContainer;

        protected:
          ObjRefContainer        m_objectsToAdd;
          ObjRefContainer        m_objectsToRemove;
          agx::Mutex         m_mutex;
      };

      template< typename T >
      class AddRemoveEnableCompatibleObject : public AddRemoveCompatibleObject< T >
      {
        public:
          AddRemoveEnableCompatibleObject() {}

          void setEnable( T* object, bool enable )
          {
            if ( object == nullptr )
              return;

            agx::ScopeLock< agx::Mutex > sl( m_mutex );

            if ( enable ) {
              if ( m_objectsToEnable.contains( object ) )
                return;
              m_objectsToEnable.push_back( object );
              m_objectsToDisable.findAndErase( object );
            }
            else {
              if ( m_objectsToDisable.contains( object ) )
                return;
              m_objectsToDisable.push_back( object );
              m_objectsToEnable.findAndErase( object );
            }
          }

          void commit( agxSDK::Simulation* simulation )
          {
            AddRemoveCompatibleObject< T >::commit( simulation );

            while ( !m_objectsToEnable.empty() ) {
              m_objectsToEnable.back()->setEnable( true );
              m_objectsToEnable.pop_back();
            }

            while ( !m_objectsToDisable.empty() ) {
              m_objectsToDisable.back()->setEnable( false );
              m_objectsToDisable.pop_back();
            }
          }

          bool empty() const { return AddRemoveCompatibleObject<T>::empty() && m_objectsToEnable.empty() && m_objectsToDisable.empty(); }

        protected:
          typedef agx::Vector< agx::ref_ptr< T > > ObjRefContainer;

        protected:
          ObjRefContainer        m_objectsToEnable;
          ObjRefContainer        m_objectsToDisable;
          agx::Mutex             m_mutex;
      };

    protected:
      using GeometryRefIdContainer = agx::Vector< std::pair<agxCollide::GeometryRef, agx::UInt32> >;
      using GeometryRefNameContainer = agx::Vector< std::pair<agxCollide::GeometryRef, agx::Name> >;

    protected:
      agx::observer_ptr< agxSDK::Simulation >                   m_simulation;
      bool                                                      m_runningParallel;
      AddRemoveEnableCompatibleObject< agxCollide::Geometry >   m_geometries;
      AddRemoveEnableCompatibleObject< agx::RigidBody >         m_rigidBodies;
      AddRemoveEnableCompatibleObject< agx::Constraint >        m_constraints;
      AddRemoveEnableCompatibleObject< agx::StrongInteraction > m_strongInteractions;
      AddRemoveCompatibleObject< agxSDK::Assembly >             m_asssemblies;
      AddRemoveEnableCompatibleObject< agxSDK::EventListener >  m_eventListeners;
      GeometryRefIdContainer                                    m_geometryGroupsAdd;
      GeometryRefIdContainer                                    m_geometryGroupsRemove;
      GeometryRefNameContainer                                  m_namedGeometryGroupsAdd;
      GeometryRefNameContainer                                  m_namedGeometryGroupsRemove;
      agxCollide::LocalGeometryContactVector                    m_localGeometryContacts;
      agx::ContactMaterialRefVector                                  m_contactMaterialsToDefaultStorage;
      agx::ContactMaterialRefVector                                  m_contactMaterialsToSimulationStorage;
      agx::Mutex                                                m_mutex;
  };
}

DOXYGEN_END_INTERNAL_BLOCK()
