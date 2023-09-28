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

#include <agx/config/AGX_USE_WEBSOCKETS.h>
#include <agx/config.h>
#if AGX_USE_WEBSOCKETS()

#include <agxNet/SocketTrack.h>
#include <agxNet/WebSocket.h>

namespace agxNet
{

  AGX_DECLARE_POINTER_TYPES(WebSocketFrameWriter);
  class AGXCORE_EXPORT WebSocketFrameWriter : public agxData::FrameWriter
  {
  public:
    explicit WebSocketFrameWriter(agx::UInt16 port);
    explicit WebSocketFrameWriter(agxNet::WebSocket* client);

    virtual void writeFrame(const agxData::Frame* frame);

    agx::UInt16 getPort() const;

    bool service(agx::Real timeoutMs = 0);

  protected:
    virtual ~WebSocketFrameWriter() {}

    void clientConnectedCallback(agxNet::WebSocket* client);
    void clientDisconnectedCallback(agxNet::WebSocket* client);
    void messageReceivedCallback(agxNet::WebSocket *socket, agx::UInt8* payload, size_t size);

  private:
    agxNet::WebSocketServerRef m_serverSocket;
    agxNet::WebSocketPtrVector m_clientSockets;
    agx::Mutex m_mutex;
  };



  AGX_DECLARE_POINTER_TYPES(WebSocketFrameReader);
  class AGXCORE_EXPORT WebSocketFrameReader : public agxData::FrameReader
  {
  public:
    WebSocketFrameReader(agxNet::WebSocket* socket, bool active = false);

    void messageReceivedCallback(agxNet::WebSocket *socket, agx::UInt8* payload, size_t size);
    void releaseBlock(agxNet::WebSocket*);
    
    virtual agxData::Frame* readFrame();

  protected:
    virtual ~WebSocketFrameReader();

  private:
    agxNet::WebSocket* m_socket;
    agx::Block m_frameBlock;

    // Cannot use reference counting pointers here since 'readFrame' is required to
    // return Frames by raw pointer, and a vector of reference counted pointers would
    // then delete the Frames before they are passed to the caller.
    agx::Vector<agxData::Frame*> m_storedFrames;
  };


  /* Implementation */
  AGX_FORCE_INLINE agx::UInt16 WebSocketFrameWriter::getPort() const { return m_serverSocket->getPort(); }

}


// AGX_USE_WEBSOCKETS
#endif
