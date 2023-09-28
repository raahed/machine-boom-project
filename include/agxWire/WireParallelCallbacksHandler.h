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

#include <agx/Referenced.h>
#include <agx/Task.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxSDK
{
  class Simulation;
}

namespace agxWire
{
  class Wire;

  /**
  Object to handle various types of callbacks to wires. Serial and/or parallel.
  */
  class AGXPHYSICS_EXPORT WireParallelCallbacksHandler : public agx::Referenced
  {
    public:
      /**
      Supported callback types.
      */
      enum CallbackTypes
      {
        PRE_COLLIDE,
        PRE,
        POST,
        COLLIDE,
        COLLECT_SPLIT_INFO,
        NUM_CALLBACKS
      };

    public:
      /**
      Simulation specific.
      */
      WireParallelCallbacksHandler( agxSDK::Simulation* simulation );

      /**
      Initializes and resets (prepares) callback kernels.
      */
      virtual void reset( agxSDK::Simulation* simulation );

      /**
      Execute parallel- or serial mode, ends with call to WireHandler for this simulation.
      */
      virtual void execute( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::Simulation* simulation );

      DOXYGEN_START_INTERNAL_BLOCK()
      /**

      Forces execute to be serial. Used in tests.
      */
      void setExecuteSerial( agx::Bool executeSerial );

      /**

      \return true if callbacks are executed in serial mode - otherwise false
      */
      agx::Bool getExecuteSerial() const;
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      typedef agx::Vector< agx::ref_ptr< agxWire::Wire > > WireContainer;

    protected:
      virtual ~WireParallelCallbacksHandler() {}

      /**
      Execute a callback of supported type.
      */
      virtual void executeParallel( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::Simulation* simulation );

      /**
      Execute callbacks serial.
      */
      virtual void executeSerial( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::Simulation* simulation );

      /**
      Handle simplified wires. Call this method before you collect active wires!
      */
      virtual void handleSimplified( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::Simulation* simulation );

      /**
      Handle links.
      */
      virtual void handleLinks( agxWire::WireParallelCallbacksHandler::CallbackTypes type, agxSDK::Simulation* simulation );

      /**
      Prepare all dependencies and returns the list of wires belonging to \p simulation.
      */
      WireContainer prepareAndCollectWires( WireParallelCallbacksHandler::CallbackTypes type, agxSDK::Simulation* simulation );

    protected:
      agxData::BufferRef m_wireDataBuffer;
      agx::TaskRef       m_executeCallback[ NUM_CALLBACKS ];
      agx::Bool          m_executeSerial;
  };

  typedef agx::ref_ptr< WireParallelCallbacksHandler > WireParallelCallbacksHandlerRef;

  class AGXPHYSICS_EXPORT WireParallelCallbacksHandlerFactory : public agx::Referenced
  {
    public:
      WireParallelCallbacksHandlerFactory();
      virtual agxWire::WireParallelCallbacksHandler* create( agxSDK::Simulation* simulation ) const;

    protected:
      virtual ~WireParallelCallbacksHandlerFactory() {}
  };

  typedef agx::ref_ptr< WireParallelCallbacksHandlerFactory > WireParallelCallbacksHandlerFactoryRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
