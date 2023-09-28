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


#ifndef AGX_REMOTESOLVER_H
#define AGX_REMOTESOLVER_H

#include <agx/config/AGX_USE_WEBSOCKETS.h>
#include <agx/Component.h>
#include <agxNet/Socket.h>
#include <agxNet/WebSocket.h>
#include <agxSDK/SimulationTrack.FrameReader.h>
#include <agxSDK/SimulationTrack.FrameWriter.h>

#if AGX_USE_WEBSOCKETS()

#include <agxData/Frame.h>

namespace agxSDK { class Simulation; }

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(RemoteSolver);
  class AGXPHYSICS_EXPORT RemoteSolver : public Referenced
  {

  public:
    RemoteSolver(UInt16 tcpSocketPort = 1906, UInt16 webSocketPort = 1907);


    void execute();

    agx::UInt16 getWebPort() const { return m_webPort; }
    agx::UInt16 getTcpPort() const { return m_tcpPort; }
    agxSDK::SimulationFrameReader *getSimulationReader() { return m_simulationReader; }
    agxSDK::SimulationFrameWriter *getSimulationWriter() { return m_simulationWriter; }

  protected:
    virtual ~RemoteSolver();

  private:
    friend class agxSDK::Simulation;
    void init(agxSDK::Simulation *simulation);

  private:
    void clientConnected(agxNet::WebSocket* client);
    void clientConnected(agxNet::TCPSocket* client);
    void clientDisconnected(agxNet::WebSocket* client);
    void clientDisconnected(agxNet::TCPSocket* client);
    void messageReceived(agx::UInt8* payload, size_t size);

    void waitForClient();
    void someClientDisconnected();
    bool writeReadCycle(agxData::Frame* frame);

  private:
    agx::UInt16 m_tcpPort;
    agx::UInt16 m_webPort;
    agxSDK::Simulation *m_simulation;

    agxSDK::SimulationFrameReaderRef m_simulationReader;
    agxSDK::SimulationFrameWriterRef m_simulationWriter;

    agxNet::TCPServerSocketRef m_tcpSocketServer;
    agxNet::TCPSocketRef m_tcpSocketClient;

    agxNet::WebSocketServerRef m_webSocketServer;
    agxNet::WebSocketRef m_webSocketClient;

    agxData::FrameChannelRef m_frameChannel;

    agxData::FrameRef m_solverResult;
  };

}
DOXYGEN_END_INTERNAL_BLOCK()

#endif
#endif /* AGX_REMOTESOLVER_H */
