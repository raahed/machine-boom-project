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

#ifdef _MSC_VER
#  pragma  warning( disable : 4290)
#  pragma warning( disable: 4275 ) //  warning C4275: non dll-interface class
#endif

#include <agx/agxPhysics_export.h>
#include <agx/Thread.h>
#include <agxNet/Compress.h>
#include <agxNet/Socket.h>
#include <agx/Referenced.h>

namespace agxNet
{

  /// Base class for multi threaded transferring simulation between a server and a client using sockets
  class AGXPHYSICS_EXPORT CoSimulationBase
  {
  public:

    /// Header used for sending data over sockets
    struct Header
    {
      Header() : numBytes(0), mode(0), compressed(false)  {}
      enum Mode
      {
        NOP = 0,

        PRELOAD_SCRIPT_FILE,
        PORT_INFORMATION,
        INIT_INPUT,
        INIT_OUTPUT,
        SET_TIME_STEP,
        STEP_INPUT,
        STEP_OUTPUT,

        ERROR_MESSAGE,
        WARNING_MESSAGE,
        OK_MESSAGE,

        SHUTDOWN
      };

      uint32_t numBytes;
      uint8_t mode;
      bool compressed;
      char md5[32];
    };

  public:

    /// Default constructor
    CoSimulationBase();

    /// \return true if we are connected to a remote party
    virtual bool isConnected();

    virtual bool waitForConnection()=0;

    /**
     Drop the connection to the remote host and wait for a new one.
     \param shouldShutdown - Can be used by derived classes to shutdown the communication.
     */
    virtual void dropConnection(bool shouldShutdown=false);

  protected:

    /**
    Transmit data to remote party. Data block can be chopped up into smaller chunks.
    \param data - bit stream that should be transmitted to remote client
    \param numBytes - number of bytes to be transmitted.
    \return number of bytes sent
    */
    size_t sendData( const uint8_t* data, size_t numBytes );

    /**
    Receive data from remote party.
    \param data - pointer to address where data will be written
    \param numBytes - size of memory storage.
    \return number of bytes read.
    */
    size_t receiveData( uint8_t* data, size_t numBytes );

    virtual ~CoSimulationBase();

  protected:

    virtual void handleConnectionDrop();
    void sendPacket(Header &header);
    void receivePacket(Header &header);

    agx::Vector<char> m_buffer;

    agx::Mutex m_connectionMutex;
    TCPSocketRef m_socket;
    bool m_isShutdown;


  private:

  };


  template<typename T>
  void insertBytes(agx::Vector<char>& container, size_t startIndex, const T& value)
  {
    ::memcpy(&container[startIndex], &value, sizeof(value));
  }


  template<typename T>
  T extractBytes(const agx::Vector<char>& container, size_t startIndex)
  {
    T value;
    ::memcpy(&value, &container[startIndex], sizeof(T));
    return value;
  }

}
