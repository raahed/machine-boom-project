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

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning( disable: 4251 ) //   'agxNet::RemoteDebugClient::m_buffer' : class 'agx::Vector<T>' needs to have dll-interface to be used by clients of class 'agxNet::RemoteDebugClient'
#endif

namespace agxSDK
{
  class Simulation;
}

namespace agxNet
{

  /// Class for receiving serialized agxSDK::Simulation:s from remote host
  class AGXPHYSICS_EXPORT RemoteDebugClient : public agxNet::RemoteDebugBase
  {
  public:
    /**
    Constructor.
    \param address - Address of server running a simulation
    */
    RemoteDebugClient( const IPAddress& address );

    /**
    Take a received Simulation, and add the data into the specified simulation.
    \param simulation - The simulation where data will be inserted
    */
    void receiveSimulation( agxSDK::Simulation *simulation );

    /// Is the client still waiting for connection?
    bool isWaitingForConnection() const;

  protected:

    bool connectToServer();

    virtual void waitForDataAndTransfer();
    virtual bool waitForConnection();
    virtual void handleConnectionDrop();

    virtual ~RemoteDebugClient();

  private:
    IPAddress m_address;
    agx::Vector<char> m_buffer;
    bool m_isWaitingForConnection;


  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

