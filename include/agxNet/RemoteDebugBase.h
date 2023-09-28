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
#  pragma warning(push)
#  pragma  warning( disable : 4290)
#  pragma warning( disable: 4275 ) //  warning C4275: non dll-interface class
#endif

#include <agx/agxPhysics_export.h>
#include <agx/Thread.h>
#include <agxNet/Compress.h>
#include <agxNet/Socket.h>
#include <agx/Referenced.h>
#include <agx/Thread.h>
#include <agx/List.h>

namespace agxNet
{

  /// Base class for multi threaded transferring simulation between a server and a client using sockets
  class AGXPHYSICS_EXPORT RemoteDebugBase : public agx::BasicThread, public agx::Referenced
  {
  public:

    /// Header used for sending data over sockets
    struct Header
    {
      Header() : numBytes(0), mode(0), compressed(false)  {}
      enum Mode
      {
        READY_TO_RECEIVE=121,
        NEW_FRAME=111
      };

      uint32_t numBytes;
      uint8_t mode;
      bool compressed;
      char md5[32];
    };

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma  warning( disable : 4250) // 'agxNet::RemoteDebugBase::DataBlock' : inherits 'std::basic_istream<_Elem,_Traits>::std::basic_istream<_Elem,_Traits>::_Add_vtordisp1' via dominance
#endif


    struct DataBlock : public agx::Referenced, public std::stringstream
    {
    };
    typedef agx::ref_ptr<DataBlock> DataBlockRef;

#ifdef _MSC_VER
# pragma warning(pop)
#endif

  public:

    /// Default constructor
    RemoteDebugBase();

    /**
    Push a block of data for access either by client (simulation) or to be transmitted over socket.
    \param data - block of bit stream data to be transmitted over the net
    */
    void pushBlock( DataBlock *data  );

    /**
    \return 0 if no block left, otherwise a reference to a block of data ready to be transmitted/read
    */
    DataBlockRef getBlock(  );
    void run();

    /// \return true if we are connected to a remote party
    virtual bool isConnected();

    /// \return true if we have data in the pipeline (someone has done pushBlock())
    bool hasData();

    /// \param numElements - specifies the size of the queue of unprocessed elements. For 1 we do not store any queue at all.
    void setMaxQueueSize( size_t numElements ) { m_queueSize = numElements; }

    /// \return the specified maximum queue size
    size_t getMaxQueueSize( ) const { return m_queueSize; }

    /// \return number of elements in the queue waiting for processing.
    size_t getQueueSize();

    /// Shutdown the communication threads, will cause this client to be non-usable any more.
    void shutdown();

  protected:


    bool isShutdown();

    virtual void waitForDataAndTransfer()=0;
    virtual bool waitForConnection()=0;

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

    virtual ~RemoteDebugBase();

    /**
    Drop the connection to the remote host and wait for a new one.
    */
    virtual void dropConnection();

  protected:

    void waitForData();
    void releaseData();
    virtual void handleConnectionDrop();

    agx::Block m_hasData;
    agx::Block m_hasReader;

    agx::Mutex m_mutex;
    TCPSocketRef m_socket;
    size_t m_queueSize;
    bool m_isShutdown;


  private:
    typedef agx::List< agx::ref_ptr<DataBlock> > DataQueue;
    DataQueue m_queue;

  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

