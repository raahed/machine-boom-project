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

//#include <signal.h>
#pragma once

#include <agx/config/AGX_USE_WEBSOCKETS.h>

#if AGX_USE_WEBSOCKETS()

#include <agxNet/Socket.h>
#include <agxNet/WebSocket.h>
#include <agxNet/FramePacket.h>
#include <agxData/Track.h>
#include <agxNet/WebSocketTrack.h>
#include <agxSDK/SimulationTrack.FrameWriter.h>
#include <agx/Thread.h>

//#include <agx/Interrupt.h>
//#include <agxIO/ArgumentParser.h>

#include <agxSDK/Simulation.h>

using namespace agx;
using namespace agxSDK;

namespace agxNet{
  AGX_DECLARE_POINTER_TYPES(RemoteJournalConnector);
  AGX_DECLARE_VECTOR_TYPES(RemoteJournalConnector);

  /**
   * A base class for connecting to remote journals over websockets.
   */
  class AGXPHYSICS_EXPORT RemoteJournalConnector {
    public:
      /*
       * Initialize a RemoteJournalConnector entity that expects a
       * control channel server at host:port
       */
      RemoteJournalConnector(String host, UInt16 port);
      virtual ~RemoteJournalConnector();

      /*
       * Attempts to connect to a control channel at host:port
       */
      void connectToControlChannel();
      agxSDK::Simulation* getSimulation();

      /*
       * Sends a "Simulation.Play" message to the control server
       */
      void sendPlayMessage();

      /*
       * Mutex that gets locked during manipulation of the Simulation
       */
      virtual agx::Mutex& getFrameMutex()
      {
        static agx::Mutex s_defaultMutex;
        return s_defaultMutex;
      };

      /*
      \return True as long as the connection should be kept
      */
      virtual bool isRunning() = 0;

      /*
       * Called after the control channel have recieved and parsed a "Simulation.LoadScene" message
       */
      virtual void onLoadSimulation(Simulation*) {};

      /*
       * Called after the connection to the Journal Socket is established
       */
      virtual void onRemoteViewerAttached() {};

      /*
       * Called when the control channel receives a "Simulation.Exit" message
       */
      virtual void onSimulationExit() {};

    private:
      void connectToJournalSocket();
      void handleSimulationLoadScene(agxNet::StructuredMessage* message);
      void onControlChannelMessage(agxNet::WebSocket *, UInt8 *data, size_t numBytes);
      void onDisconnected(agxNet::WebSocket* disconnectedSocket);
      std::stringstream readAgxSceneDataFromLoadSceneMessage(agxNet::StructuredMessage* message);
      void runServerSocketThread();
      bool service();

      agxSDK::SimulationRef m_simulation;
      agxNet::WebSocketServerRef m_serverSocket;
      agxNet::WebSocketRef m_controlChannelSocket;
      agxNet::WebSocketRef m_journalTrackSocket;
      agxNet::WebSocketFrameReaderRef m_journalTrackReader;
      agxSDK::SimulationFrameWriterRef m_simulationWriter;
      agxData::TrackRef m_socketToSimulationTrack;
      String m_host;
      UInt16 m_controlChannelPort;
      UInt16 m_journalSocketPort;
      std::atomic<bool> m_serverSocketShouldStop;
      std::thread m_serverSocketThread;
  };
}

#endif
