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

#ifndef AGXOSG_VIDEOSTREAM_H
#define AGXOSG_VIDEOSTREAM_H

#include <agxOSG/RenderToTexture.h>
#include <agxOSG/VideoCapture.h>
#include <agxNet/WebSocket.h>

#if AGX_USE_FFMPEG()

namespace agxOSG
{
  class ExampleApplication;

  AGX_DECLARE_POINTER_TYPES(VideoStream);
  class VideoStream : public agx::Referenced
  {
  public:
    VideoStream(ExampleApplication *app, agx::UInt16 port, agx::UInt width, agx::UInt height);

    bool service();

    agx::UInt16 getPort() const;

  protected:
    virtual ~VideoStream();

  private:

    void clientConnected(agxNet::WebSocket *socket);
    void clientDisconnected(agxNet::WebSocket *socket);
    void writeVideoData(agx::UInt8 *data, size_t size);
    void receiveClientData(agxNet::WebSocket *socket, agx::UInt8* data, size_t size);

  private:
    agxNet::WebSocketServerRef m_server;
    agxNet::WebSocketServer::SocketCallback m_socketConnectedCallback;
    agxNet::WebSocketServer::SocketCallback m_socketDisconnectedCallback;
    agxNet::WebSocket::ReceiveCallback m_socketReceiveCallback;
    RenderToTextureRef m_rtt;
    VideoCaptureRef m_capture;
    ExampleApplication *m_app;
    agx::Mutex m_mutex;
    agx::UInt m_width;
    agx::UInt m_height;
    agx::Real m_eventMaxX;
    agx::Real m_eventMaxY;
  };

}

#endif
#endif /* AGXOSG_VIDEOSTREAM_H */
