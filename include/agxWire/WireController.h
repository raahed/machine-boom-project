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

#include <agxWire/Wire.h>
#include <agxWire/WireHandler.h>
#include <agxWire/WireParallelCallbacksHandler.h>

#include <agxSDK/Simulation.h>

#include <agx/agxPhysics_export.h>
#include <agx/Singleton.h>

namespace agxWire
{
  /**
  Overlaying object handling wires. This class monitors construction and deletion
  (and adding, removing from simulations) of wires and enables interface for
  enabling and disabling contacts/collision handling between wires.

  Note that this class handles agxWire::Wire objects, and nothing else.

  Examples how to use this class.

  Example 1:
  agxWire::WireRef wire1 = new ...
  agxWire::WireRef wire2 = new ...
  agxWire::WireController::instance()->getEnableCollisions( wire1, wire2 ); // false - default is false.
  agxWire::WireController::instance()->setEnableCollisions( wire1, wire2, true );

  Example 2:
  // Loop ALL wire objects created (registered during "new agxWire::Wire").
  agxWire::WireController::WireIterator it = agxWire::WireController::instance()->getBeginWireIterator();
  while ( it != agxWire::WireController::instance()->getEndWireIterator() ) {
    const agxWire::Wire* wire = *it;
    // Do stuff.
    ++it;
  }
  */
  class AGXPHYSICS_EXPORT WireController : public agx::Singleton
  {
    public:
      typedef agx::SetVector< agxWire::Wire* >                                         WirePtrContainer;
      typedef WirePtrContainer::const_iterator                                         WireIterator;
      typedef agx::SymmetricPair< agx::observer_ptr< agxWire::Wire > >                 WireWireSymmetricPair;
      typedef agx::Vector< WireWireSymmetricPair >                                     WireWireContainer;
      typedef agx::HashVector< agx::UInt32, agx::observer_ptr< agxWire::Wire > >       IdWireContainer;
      typedef agx::Vector< WireHandlerRef >                                            WireHandlerContainer;
      typedef std::pair< agxSDK::SimulationObserver, WireParallelCallbacksHandlerRef > SimulationObsWireParallelCallbacksHandlerRefPair;
      typedef std::pair< agxSDK::SimulationObserver, WirePtrContainer >                SimulationObsWirePtrContainerPair;
      typedef agx::Vector< SimulationObsWireParallelCallbacksHandlerRefPair >          WireParallelCallbacksHandlerContainer;
      typedef agx::Vector< SimulationObsWirePtrContainerPair >                         SimplifiedWiresContainer;

   public:
      /**
      Internal state handling. Test, debug and development related.
      */
      enum State
      {
        EXPLICIT_ENABLE      = (1<<0),         /**< The user must call setEnableCollisions explicitly for all pairs of wires. */
        ALL_ENABLE           = (1<<1),         /**< Collisions enabled between all pairs of wires. */
        USE_DYNAMIC_CONTACTS = (1<<2),         /**< Make all wire contacts dynamic. */
        DEFAULT              = EXPLICIT_ENABLE
      };

    public:
      /**
      \return instance of the object handling wire-wire collisions
      */
      static WireController* instance();

      /**
      \returns true if the wire-wire framework is active (i.e., collisions enabled
               between at least one wire pair or state is ALL_ENABLE) - otherwise false
      */
      bool isWireWireActive() const;

      /**
      Enable, or disable, collision detection and contacts between two wires.
      By default collisions are disabled.
      \param wire1 - first wire
      \param wire2 - second wire
      \param enable - true to enable collisions between \p wire1 and \p wire2 - false to disable
      \return true if the new state was accepted - otherwise false
      */
      bool setEnableCollisions( agxWire::Wire* wire1, agxWire::Wire* wire2, bool enable );

      /**
      \param wire1 - first wire
      \param wire2 - second wire
      \return true if \p wire1 can collide with \p wire2 - otherwise false
      */
      bool getEnableCollisions( const agxWire::Wire* wire1, const agxWire::Wire* wire2 ) const;

      /**
      Enable, or disable, the use of dynamic wire contact model against this geometry.
      \note This method will add the WIRE_DYNAMIC_CONTACTS_GROUP group to the geometry,
            and contacts between geometries with the WIRE_DYNAMIC_CONTACTS_GROUP group
            and wire lump spheres - are by definition disabled.
      \note This geometry will not be 'remembered' after this call. I.e., the changes
            made to the geometry by calling this method with enable == true can only
            be reverted when this method is called with the same geometry and enable == false.
      \param geometry - geometry to change wire contact model on
      \param enable - true to use the dynamics contact model, false to use the default contact model
      \return true if the desired state was accepted / successful - otherwise false
      */
      agx::Bool setEnableDynamicWireContacts( agxCollide::Geometry* geometry, agx::Bool enable );

      /**
      Enable, or disable, the use of dynamic wire contact model against this rigid body.
      \note This method will add the WIRE_DYNAMIC_CONTACTS_GROUP group to the geometries
            in the given rigid body, and contacts between geometries with the
            WIRE_DYNAMIC_CONTACTS_GROUP group and wire lump spheres - are by definition disabled.
      \note The geometries will not be 'remembered' after this call. I.e., the changes
            made to the geometry by calling this method with enable == true can only
            be reverted when this method is called with the same geometry and enable == false.
      \param rb - rigid body with geometries to change wire contact model on
      \param enable - true to use the dynamics contact model, false to use the default contact model
      \return true if the desired state was accepted / successful - otherwise false
      */
      agx::Bool setEnableDynamicWireContacts( agx::RigidBody* rb, agx::Bool enable );

      /**
      Force all wire contacts to be dynamic.
      */
      void setEnableDynamicWireContactsGlobally( agx::Bool enable);

      /**
      \return true if the contact model for \p geometry is 'dynamic' - otherwise false
      */
      agx::Bool getEnableDynamicWireContacts( const agxCollide::Geometry* geometry ) const;

      /**
      \return true if all geometries in \p rb has contact model 'dynamic' - otherwise false
      */
      agx::Bool getEnableDynamicWireContacts( const agx::RigidBody* rb ) const;

      /**
      \returns true if all wire contacts are forced to be dynamic.
      */
      agx::Bool getEnableDynamicWireContactsGlobally() const;

      /**
      \return the wire-wire handler given a simulation (this handler is unique to the simulation)
      */
      agxWire::WireHandler* getOrCreateWireHandler( agxSDK::Simulation* simulation );

      /**
      \return the wire parallel callbacks handler given a simulation (this handler is unique to the simulation)
      */
      agxWire::WireParallelCallbacksHandler* getOrCreateWireParallelCallbacksHandler( agxSDK::Simulation* simulation );

      /**
      Removes object.
      */
      void removeParallelCallbackHandler( agxSDK::Simulation* simulation );

      /**
      \return wire handler given simulation (0 if none has been created)
      */
      agxWire::WireHandler* find( const agxSDK::Simulation* simulation );

      /**
      \return the simulation given a wire-wire handler (the handler is unique to the simulation)
      */
      agxSDK::Simulation* find( agxWire::WireHandler* handler ) const;

      /**
      Access to all the wires. getEndWireIterator() - getBeginWireIterator() == total number of wires.
      \return begin iterator
      */
      WireIterator getBeginWireIterator() const;

      /**
      \return end wire iterator
      */
      WireIterator getEndWireIterator() const;

      /**
      Can be used to iterate over all wires in a script language that e.g., not support iterators.
      \return the number of wire this wire controller holds
      */
      agx::UInt getNumWires() const;

      /**
      \return wire with index \p index
      */
      agxWire::Wire* getWire( agx::UInt index ) const;

      /**
      Internal method. Pairs of wire where setEnableCollisions( ..., true ) has been called.
      */
      const WireWireContainer& getEnabledWireWirePairs() const;

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Internal method.
      \return a wire given its id - 0 if the wire isn't part of wire-wire contacts
      */
      agxWire::Wire* findWire(agx::UInt32 id) const;

      /**
      Internal method.
      \return a wire given a geometry (given unique id) - 0 if nothing were found
      */
      agxWire::Wire* findWire(const agxCollide::Geometry* geometry) const;

      /**
      Internal method. Simulation specific wire handler.
      */
      bool add(agxSDK::Simulation* simulation, agxWire::WireHandler* wireHandler);

      /**
      Internal method. Remove the simulation specific wire-wire handler. Wire-wire contacts will be delayed or disabled.
      */
      bool remove(agxWire::WireHandler* wireHandler);

      /**
      Internal method. The current, global state mask.
      */
      agx::UInt getStateMask() const;

      /**
      Internal method. Assign the current, global state mask.
      */
      void setStateMask( agx::UInt stateMask );

      /**
      Add global state.
      \param state - state to add
      */
      void addState( State state );

      /**
      Remove global state.
      \param state - state to remove
      */
      void removeState( State state );

      /**
      \param state - state to check
      \return true if state is active - otherwise false
      */
      agx::Bool hasState( State state ) const;

      /**
      \return id -> wire table
      */
      const agxWire::WireController::IdWireContainer& getIdWireContainer() const;

      /**
      \return container with simplified wires given the simulation they were part of
      */
      agxWire::WireController::WirePtrContainer getSimplifiedWires(const agxSDK::Simulation* simulation) const;

      /**
      Register new parallel handler factory.
      */
      void registerParallelHandlerFactory(agxWire::WireParallelCallbacksHandlerFactory* factory);

      /**
      \return the current parallel handler factory
      */
      agxWire::WireParallelCallbacksHandlerFactory* getParallelHandlerFactory() const;

      /**
      \return spin mutex for critical sections running in parallel
      */
      agx::ReentrantMutex& getMutex();

      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      \return all created wires
      */
      const agxWire::WireController::WirePtrContainer& getAllWires() const;


    protected:
      WireController();
      virtual ~WireController();

      virtual void shutdown() override;

      friend class Wire;
      friend class WireGeometryController;
      friend class WireSimplifyController;

      /**
      Internal method. All wires should report themselves to this controller during construction.
      */
      void onCreate( agxWire::Wire* wire );

      /**
      Internal method. All wires should report themselves to this controller during addNotification.
      */
      void onAddNotification( agxWire::Wire* wire );

      /**
      Internal method. All wires must be removed from the controller when the destructor is called.
      */
      void onDelete( agxWire::Wire* wire );

      /**
      Internal method. From removeNotification.
      */
      void onRemoveNotification( agxWire::Wire* wire );

      /**
      Internal method. Updates identification number for a wire.
      */
      void updateId( agxWire::Wire* wire );

      /**
      Call when the wire is simplified.
      */
      void onSimplify( agxSDK::Simulation* simulation, agxWire::Wire* wire );

      /**
      Call when the wire is unsimplified.
      */
      void onUnsimplify( agxSDK::Simulation* simulation, agxWire::Wire* wire );

      SINGLETON_CLASSNAME_METHOD();

    protected:
      WirePtrContainer                       m_wires;                     /**< All wires instantiated. */
      WireWireContainer                      m_enabledWires;              /**< Explicitly enabled collisions pairs. */
      IdWireContainer                        m_idWire;                    /**< Wire unique geometry id -> wire. */
      WireHandlerContainer                   m_wireHandlers;              /**< Wire handlers, unique for each simulation. */
      WireParallelCallbacksHandlerFactoryRef m_parallelCallbacksFactory;
      WireParallelCallbacksHandlerContainer  m_parallelCallbacksHandlers; /**< Parallel callbacks handler, unique for each simulation. */
      SimplifiedWiresContainer               m_simplifiedWires;
      agx::UInt                              m_state;                     /**< Current global state. */
      agx::ReentrantMutex                    m_mutex;
    private:
      static WireController*   s_instance;
  };

  AGX_FORCE_INLINE WireController::WireIterator WireController::getBeginWireIterator() const
  {
    return m_wires.begin();
  }

  AGX_FORCE_INLINE WireController::WireIterator WireController::getEndWireIterator() const
  {
    return m_wires.end();
  }
}

