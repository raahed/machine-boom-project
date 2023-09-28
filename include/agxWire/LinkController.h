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


#include <agxWire/Link.h>

#include <agx/Singleton.h>
#include <agx/SetVector.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxSDK
{
  class SimulationProxy;
}

namespace agxWire
{
  class AGXPHYSICS_EXPORT LinkController : public agx::Singleton
  {
    public:
      typedef agx::Vector< agxWire::LinkRef > LinkRefContainer;

    public:
      /**
      \return instance of the object handling links
      */
      static agxWire::LinkController* instance();

      /**
      \return true if this controller has been instantiated, otherwise false
      */
      static agx::Bool hasInstance();

    public:
      /**
      Global search for the links connected to the wire.
      \return links connected to the wire
      */
      agxWire::LinkController::LinkRefContainer getLinks( const agxWire::Wire* wire ) const;

    public:
      /**
      Internal method. Handle callbacks for links in a given simulation.
      */
      void update( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::SimulationProxy* simulationProxy );

      /**
      Internal method. Wire is being removed from a simulation.
      */
      void onRemoveNotification( agxWire::Wire* wire );

    protected:
      LinkController();
      virtual ~LinkController();

      virtual void shutdown() override;

      void refCount( agxWire::LinkRef link, agx::Int32 count );

      friend class Link;
      void onCreate( agxWire::Link* link );
      void onDelete( agxWire::Link* link );

      void onConnect( agxWire::LinkRef link );
      void onDisconnect( agxWire::LinkRef link );

      SINGLETON_CLASSNAME_METHOD();

    protected:
      typedef agx::SetVector< agxWire::Link* > LinkContainer;
      typedef agx::HashTable< agxWire::LinkRef, agx::UInt > LinkRefCountContainer;

    protected:
      LinkContainer         m_links;
      LinkRefCountContainer m_linksRefCounter;

    private:
      static LinkController* s_instance;
  };
}

DOXYGEN_END_INTERNAL_BLOCK()
