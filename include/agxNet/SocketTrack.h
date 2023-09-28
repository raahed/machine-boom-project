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

#include <agxData/FrameIO.h>
#include <agxNet/Socket.h>
#include <agxNet/FramePacket.h>
#include <agx/Json.h>



namespace agxNet
{
  /**
   * A FrameWriter that sends frames on a ordinary streaming TCP socket.
   */
  AGX_DECLARE_POINTER_TYPES(TcpSocketFrameWriter);
  class AGXCORE_EXPORT TcpSocketFrameWriter : public agxData::FrameWriter
  {
  public:

    /**
     * Use this constructor when you want the TcpSocketFrameWriter to open a server socket.
     */
    explicit TcpSocketFrameWriter(agx::UInt16 port);

    /*
     * Use this constructor when you already have a socket you want to send frames through.
     */
    explicit TcpSocketFrameWriter(agxNet::TCPSocket* socket);

    virtual void writeFrame(const agxData::Frame* frame);

    virtual void endOfStream();

    agx::UInt16 getPort() const;
  protected:
    virtual ~TcpSocketFrameWriter();

  private:
    void acceptIncommingConnectionRequests();
    void sendPacket(agxNet::FramePacket *packet, agxNet::TCPSocket* socket);
    void sendPadding(size_t numBytes, agxNet::TCPSocket* socket);

  private:
    agx::UInt16 m_port;
    agxNet::TCPServerSocketRef m_serverSocket;
    agxNet::TCPSocketRefVector m_clientSockets;
  };



  AGX_DECLARE_POINTER_TYPES(TcpSocketFrameReader);
  class AGXCORE_EXPORT TcpSocketFrameReader : public agxData::FrameReader
  {
  public:
    TcpSocketFrameReader(agxNet::IPAddress address);
    TcpSocketFrameReader(agxNet::TCPSocket* socket);

    virtual agxData::Frame* readFrame();

    virtual bool jumpToFrame(agx::UInt frameIndex);

  protected:
    virtual ~TcpSocketFrameReader() {}

  private:
    bool readHeader();
    size_t sumBufferSizes(agxJson::Value& header);
    void readPadding(size_t padding);

  private:
    agxNet::TCPSocketRef m_socket;
    agxJson::Value m_nextHeader;
    agxData::BinaryDataRef m_binarySegment;
    StructuredMessage::PreHeader m_preHeader;

    bool m_isValid;
  };


  ////// Implementation
  AGX_FORCE_INLINE bool TcpSocketFrameReader::jumpToFrame(agx::UInt /*frameIndex*/) { return false; }
  AGX_FORCE_INLINE agx::UInt16 TcpSocketFrameWriter::getPort() const { return m_port; }
}
