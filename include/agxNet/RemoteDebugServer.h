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

#include <agxNet/RemoteDebugBase.h>

namespace agxNet
{

  /// Class for sending serialized agxSDK::Simulation over to a client
  class RemoteDebugServer : public agxNet::RemoteDebugBase
  {
  public:

    /**
    Constructor.
    \param port - The port where the server will listen for connections.
    */
    RemoteDebugServer( uint16_t port );

    /**
    Enable/disable compression of transmitted data
    \param flag - if true, data will be compressed
    */
    void setEnableCompression(bool flag);

    /**
    \return true if compression is enabled
    */
    bool getEnableCompression() const;


    /**
    \return true if server is waiting for client to send ready to recieve before sending new package
    */
    bool getEnableFrameSynchronizaton() const;

    /**
    Specifies if server should wait for client to send ready to recieve before sending new package.
    By default this is set to true.
    If set to false, server will constantly send new frames no matter if a client is waiting or not.
    */
    void setEnableFrameSyncronization(bool flag);


  protected:

    virtual void waitForDataAndTransfer();
    virtual bool waitForConnection();

    virtual ~RemoteDebugServer();

  private:
    TCPServerSocketRef m_serverSocket;
    uint16_t m_port;
    bool m_useCompression;
    agx::Vector<char> m_buffer;
    bool m_enableFrameSyncronization;

  };

}
