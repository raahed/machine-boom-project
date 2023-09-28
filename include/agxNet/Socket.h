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

/* The MIT License:

Copyright (c) 2008 Ivan Gagis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. */

// (c) Ivan Gagis 2008
// e-mail: igagis\gmail.com

// Description:
//          cross platform C++ Sockets wrapper
//


#pragma once

#ifdef _MSC_VER
# pragma  warning( disable : 4290)
# pragma warning( disable: 4275 )  //  warning C4275: non dll-interface class
#endif


#include <exception>
#include <stdexcept>
#include <new>
#include <agxNet/FramePacket.h>
#include <agx/stdint.h>
#include <agx/agxCore_export.h>
#include <agx/Singleton.h>
#include <agx/Referenced.h>
#include <agx/Vector.h>
#include <agx/String.h>
#include <iosfwd>

namespace agxNet{

  //forward declarations
  class SocketSet;
  class IPAddress;
  class Socket;

  /**
  \brief Basic exception class.
  This is a basic exception class of the library. All other exception classes are derived from it.
  */
  class AGXCORE_EXPORT NetError : public std::runtime_error {
  public:
    /**
    \brief Exception constructor.
    \param message Pointer to the exception message null-terminated string. Constructor will copy the string into objects internal memory buffer.
    */
    NetError(const char* message) : std::runtime_error(message), m_socket(nullptr) {}
    NetError(Socket *socket) : std::runtime_error("Connection closed"), m_socket(socket) {}
    NetError(const char* message, Socket* socket) : std::runtime_error(message), m_socket(socket) {}

    virtual ~NetError() throw() {}

    Socket *getSocket() { return m_socket; }
  private:
    Socket *m_socket;
  };

  /**
  Indicates a timeout between server/client.
  */
  class AGXCORE_EXPORT TimeoutConnectError : public NetError {
  public:
    /**
    \brief Exception constructor.
    \param message Pointer to the exception message null-terminated string. Constructor will copy the string into objects internal memory buffer.
    */
    TimeoutConnectError(const char* message) : NetError(message) {}
    virtual ~TimeoutConnectError() noexcept {}
  };

  /**
  \return An available network port.
  */
  agx::UInt16 AGXCORE_EXPORT getAvailablePort();

  /**
  \brief a structure which holds IP address
  */
  class AGXCORE_EXPORT IPAddress
  {
  public:

    /// Default constructor
    IPAddress();

    /**
    \brief Create IP address specifying exact ip address and port number.
    \param h - IP address. For example, 0x100007f represents "127.0.0.1" IP address value.
    \param p - IP port number.
    */
    inline IPAddress(agx::UInt32 h, agx::UInt16 p) :
    m_host(h),
      m_port(p)
    {}

    /**
    \brief Create IP address specifying exact ip address as 4 bytes and port number.
    The ip address can be specified as 4 separate byte values, for example:
    \code
    IPAddress ip(127, 0, 0, 1, 80); //"127.0.0.1" port 80
    \endcode
    \param h1 - 1st triplet of IP address.
    \param h2 - 2nd triplet of IP address.
    \param h3 - 3rd triplet of IP address.
    \param h4 - 4th triplet of IP address.
    \param p - IP port number.
    */
    inline IPAddress(agx::UInt8 h1, agx::UInt8 h2, agx::UInt8 h3, agx::UInt8 h4, agx::UInt16 p) :
    m_host(agx::UInt32(h1) + (agx::UInt32(h2)<<8) + (agx::UInt32(h3)<<16) + (agx::UInt32(h4)<<24)),
      m_port(p)
    {
    }

    /**
    \brief Create IP address specifying ip address as string and port number.
    \param ip - IP address string. Example: "127.0.0.1".
    \param p - IP port number.
    */
    IPAddress(const agx::String& ip, agx::UInt16 p);

    /**
    \brief compares two IP addresses for equality.
    \param ip - IP address to compare with.
    \return true if hosts and ports of the two IP addresses are equal accordingly.
    \return false otherwise.
    */
    inline bool operator==(const IPAddress& ip){
      return (this->m_host == ip.m_host) && (this->m_port == ip.m_port);
    }

    /// \return a const reference to the host id
    const agx::UInt32& host() const { return m_host; }

    /// \return a reference to the host id
    agx::UInt32& host() { return m_host; }

    /// \return a const reference to the port
    const agx::UInt16& port() const { return m_port; }

    /// \return a reference to the port
    agx::UInt16& port() { return m_port; }

    agx::String string() const;
  private:
    //parse IP address from string
    static agx::UInt32 parseString(const char* ip);

    agx::UInt32 m_host;///< IP address
    agx::UInt16 m_port;///< IP port number

  };

  std::ostream& operator<<(std::ostream& stream, const IPAddress& adr);

  /**
  \brief Library singleton class.
  This is a library singleton class. Creating an object of this class initializes the library
  while destroying this object de initializes it. So, the convenient way of initializing the library
  is to create an object of this class on the stack. Thus, when the object goes out of scope its
  destructor will be called and the library will be de initialized automatically.
  This is what C++ RAII is all about ;-).
  */
  class AGXCORE_EXPORT Library : public agx::Singleton {

  public:
    Library();

    /**
    Get the singleton instance.
    */
    static Library *instance();

    /**
    instance the object, after this, the object cannot be used anymore.
    Called by singleton manager.
    */
    virtual void shutdown() override;

    /**
    \brief Resolve host IP by its name.
    This function resolves host IP address by its name. If it fails resolving the IP address it will Throw NetError.
    \param hostName - null-terminated string representing host name. Example: "www.somedomain.com".
    \param port - IP port number which will be placed in the resulting IPAddress structure.
    \return filled IPAddress structure.
    */
    IPAddress getHostByName(const char *hostName, agx::UInt16 port);

    SINGLETON_CLASSNAME_METHOD();

  private:

    virtual ~Library();

    void initSockets();
    void deinitSockets();

    static Library *s_instance;
    bool m_initialized;
  };

  /**
  \brief Basic socket class.
  This is a base class for all socket types such as TCP sockets or UDP sockets.
  */
  AGX_DECLARE_POINTER_TYPES(Socket);
  AGX_DECLARE_VECTOR_TYPES(Socket);
  class AGXCORE_EXPORT Socket : public agx::Referenced {
    friend class SocketSet;

  public:
    //this type will hold system specific socket handle.
    //this buffer should be large enough to hold socket handle in different systems.
    //sizeof(u64) looks enough so far (need to work on 64-bit platforms too).
#ifndef DOC_DONT_EXTRACT //direction to doxygen not to extract this class
    struct SystemIndependentSocketHandle{
      agx::UInt8 buffer[sizeof(uint64_t)]; /// \todo Will this give an alignment that works everywhere?
    };
#endif//~DOC_DONT_EXTRACT

  public:

    /**
    \brief Tells whether the socket is opened or not.
    TODO: write some detailed description.
    \return Returns true if the socket is opened or false otherwise.
    */
    bool isValid()const;

    void setBlocking(bool flag);
    bool isBlocking() const { return m_blocking; }

    /**
    \brief Tells whether there is some activity on the socket or not.
    See SocketSet class description for more info on how to use this method properly.
    \return
    - true if there is some data to read on the socket or if the remote socket has disconnected.
    The latter case can be detected by checking if subsequent Recv() method returns 0.
    - false if there is no any activity on the socket.
    */
    inline bool isReady()const
    {
      return m_readyToReceive;
    }

    void select(agx::UInt32 timeoutMillis = 0);

    inline bool readyToReceive() const
    {
      return m_readyToReceive;
    }

    inline bool readyToSend() const
    {
      return m_readyToSend;
    }

    inline const IPAddress& adr() const { return m_adr; }

    /**
    \brief Closes the socket disconnecting it if necessary.
    */
    void close();

    /**
    Returns the last reported error from a system call.
    Should be called immediately after a socket/connect call has failed.
    Typically:

        retval = ::select(int(handle + 1), &readMask, &writeMask, nullptr, &tv);
        if (retval == M_SOCKET_ERROR)
           int errorNo = Socket::getLastError();

    */
    static int getLastError();

  protected:

    /// Destructor
    virtual ~Socket(){
      this->close();
    }

    Socket(const IPAddress& adr = IPAddress());

    Socket& operator=(const Socket& s);

  protected:
    // bool m_isReady;
    bool m_readyToReceive;
    bool m_readyToSend;
    bool m_blocking;
    IPAddress m_adr;

    SystemIndependentSocketHandle m_socket;
  };


  /**
  \brief a class which represents a TCP socket.
  */
  AGX_DECLARE_POINTER_TYPES(TCPSocket);
  AGX_DECLARE_VECTOR_TYPES(TCPSocket);
  class AGXCORE_EXPORT TCPSocket : public Socket {
    friend class TCPServerSocket;
  public:
    /**
    \brief Constructs an invalid TCP socket object.
    */
    TCPSocket() {}

    /**
    \brief A copy constructor.
    Copy constructor creates a new socket object which refers to the same socket as s.
    After constructor completes the s becomes invalid.
    In other words, the behavior of copy constructor is similar to one of std::auto_ptr class from standard C++ library.
    \param s - other TCP socket to make a copy from.
    */
    //copy constructor
    TCPSocket(const TCPSocket& s, const IPAddress& adr = IPAddress()) : Socket(adr) {
      //NOTE: that operator= calls destructor, so this->socket should be invalid, base class constructor takes care about it.
      this->operator=(s);//same as auto_ptr
    }

    /**
    \brief A constructor which automatically calls TCPSocket::Open() method.
    This constructor creates a socket and calls its TCPSocket::Open() method.
    So, it creates an already opened socket.
    \param ip - IP address to 'connect to/listen on'.
    \param disableNaggle - enable/disable Naggle algorithm.
    */
    TCPSocket(const IPAddress& ip, bool disableNaggle = false) : Socket(ip) {
      this->open(ip, disableNaggle);
    }

    /**
    \brief Assignment operator, works similar to std::auto_ptr::operator=().
    After this assignment operator completes this socket object refers to the socket the s object referred, s become invalid.
    It works similar to std::auto_ptr::operator=() from standard C++ library.
    \param s - socket to assign from.
    */
    TCPSocket& operator=(const TCPSocket& s){
      this->Socket::operator=(s);
      return *this;
    }

    /**
    \brief Connects the socket.
    This method connects the socket to remote TCP server socket.
    \param ip - IP address.
    \param disableNaggle - enable/disable Naggle algorithm.
    */
    void open(const IPAddress& ip, bool disableNaggle = false);

    /**
    \brief Send data to connected socket.
    Sends data on connected socket. This method blocks until all data is completely sent.
    \param data - pointer to the buffer with data to send.
    \param size - number of bytes to send.
    \return the number of bytes sent. Note that this value should normally be equal to the size argument value.
    */
    agx::UInt32 send(const agx::UInt8* data, agx::UInt32 size);

    /**
    \brief Receive data from connected socket.
    Receives data available on the socket.
    If there is no data available this function blocks until some data arrives.
    \param buf - pointer to the buffer where to put received data.
    \param maxSize - maximal number of bytes which can be put to the buffer.
    \return if returned value is not 0 then it represents the number of bytes written to the buffer.
    \return 0 returned value indicates disconnection of remote socket.
    */
    //returns 0 if connection was closed by peer
    size_t recv(agx::UInt8* buf, size_t maxSize);

    void sendBlocking(const agx::UInt8 *data, agx::UInt32 size);
    void recvBlocking(agx::UInt8 *buf, agx::UInt32 size);

    void sendBlocking(const StructuredMessage *message);

  private:
    void disableNaggle();
  };



  /**
  \brief a class which represents a TCP server socket.
  TCP server socket is the socket which can listen for new connections
  and accept them creating an ordinary TCP socket for it.
  */
  AGX_DECLARE_POINTER_TYPES(TCPServerSocket);
  class AGXCORE_EXPORT TCPServerSocket : public Socket{
  public:
    /**
    \brief Creates an invalid (unopened) TCP server socket.
    */
    TCPServerSocket() : m_disableNaggle(false) {}

    /**
    \brief A copy constructor.
    Copy constructor creates a new socket object which refers to the same socket as s.
    After constructor completes the s becomes invalid.
    In other words, the behavior of copy constructor is similar to one of std::auto_ptr class from standard C++ library.
    \param s - other TCP socket to make a copy from.
    */
    //copy constructor
    TCPServerSocket(const TCPServerSocket& s) : Socket(), m_disableNaggle(s.m_disableNaggle)
    {
      //NOTE: that operator= calls destructor, so this->socket should be invalid, base class constructor takes care about it.
      this->operator=(s);//same as auto_ptr
    }

    /**
    \brief Assignment operator, works similar to std::auto_ptr::operator=().
    After this assignment operator completes this socket object refers to the socket the s object referred, s become invalid.
    It works similar to std::auto_ptr::operator=() from standard C++ library.
    \param s - socket to assign from.
    */
    TCPServerSocket& operator=(const TCPServerSocket& s){
      this->Socket::operator=(s);
      return *this;
    }

    /**
    \brief A constructor which automatically calls TCPServerSocket::Open() method.
    This constructor creates a socket and calls its TCPServerSocket::Open() method.
    So, it creates an already opened socket listening on the specified port.
    \param port - IP port number to listen on.
    \param disableNaggle - enable/disable Naggle algorithm for all accepted connections.
    */
    TCPServerSocket(agx::UInt16 port, bool disableNaggle = false){
      this->open(port, disableNaggle);
    }

    /**
    \brief Connects the socket or starts listening on it.
    This method starts listening on the socket for incoming connections.
    \param port - IP port number to listen on.
    \param disableNaggle - enable/disable Naggle algorithm for all accepted connections.
    */
    void open(agx::UInt16 port, bool disableNaggle = false);

    /**
    \brief Accepts one of the pending connections, non-blocking.
    Accepts one of the pending connections and returns a TCP socket object which represents
    either a valid connected socket or an invalid socket object.
    This function does not block if there is no any pending connections, it just returns invalid
    socket object in this case. One can periodically check for incoming connections by calling this method.
    \return A pointer to a TCPSocket object if a valid socket could be initialized. Returns nullptr if
    there was no any connections pending, so no connection was accepted.
    */
    TCPSocket* accept();
  private:
    bool m_disableNaggle;//this flag indicates if accepted sockets should be created with disabled Naggle

  };

  AGX_DECLARE_POINTER_TYPES(UDPSocket);
  class AGXCORE_EXPORT UDPSocket : public Socket{
  public:
    UDPSocket(){}

    /**
    \brief Open the socket.
    This method opens the socket, this socket can further be used to send or receive data.
    After the socket is opened it becomes a valid socket and Socket::IsValid() will return true for such socket.
    After the socket is closed it becomes invalid.
    In other words, a valid socket is an opened socket.
    In case of errors this method throws NetError.
    \param port - IP port number on which the socket will listen for incoming datagrams.
    This is useful for server-side sockets, for client-side sockets use UDPSocket::Open().
    */
    void open(agx::UInt16 port);


    inline void open(){
      this->open(0);
    }

    //returns number of bytes sent, should be less or equal to size.
    size_t send(const agx::UInt8* buf, agx::UInt16 size, IPAddress destinationIP);

    //returns number of bytes received, 0 if connection was gracefully closed (???).
    agx::UInt32 recv(agx::UInt8* buf, agx::UInt16 maxSize, IPAddress &out_SenderIP);

    protected:
      virtual ~UDPSocket(){
        this->close();
      }
  };


  /**
  \brief Socket set class for checking multiple sockets for activity.
  This class represents a set of sockets which can be checked for any activity
  such as incoming data received or remote socket has disconnected.
  Note, that the socket set holds only references to socket objects, so it is up to you
  to make sure that all the socket objects you add to a particular socket set will not be
  destroyed without prior removing them from socket set.
  */
  AGX_DECLARE_POINTER_TYPES(SocketSet);
  class AGXCORE_EXPORT SocketSet : public agx::Referenced {

  public:

    /**
    \brief Creates a socket set of the specified size.
    Creates a socket set which can hold the specified number of sockets at maximum.
    */
    SocketSet();

    /**
    \brief Returns number of sockets the socket set currently holds.
    \return number of sockets the socket set currently holds.
    */
    inline size_t getNumSockets()const{return m_set.size(); }

    /**
    \brief Returns maximal number of sockets the system allows.
    \return maximal number of sockets available in the system.
    */
    size_t getMaxSockets()const;

    /**
    \brief Add a socket to socket set.
    \param sock - pointer to the socket object to add.
    \return false if socket is already added
    */
    bool addSocket(Socket *sock);

    /**
    \brief Remove socket from socket set.
    \param sock - pointer to socket object which we want to remove from the set.
    \return false if socket is not previously added
    */
    bool removeSocket(Socket *sock);

    /**
    \brief Check sockets from socket set for activity.
    This method checks sockets for activities of incoming data ready or remote socket has disconnected.
    This method sets ready flag for all sockets with activity which can later be checked by
    Socket::IsReady() method. The ready flag will be cleared by subsequent TCPSocket::Recv() function call.
    \param timeoutMillis - maximum number of milliseconds to wait for socket activity to appear.
    if 0 is specified the function will not wait and will return immediately.
    \return true if there is at least one socket with activity.
    \return false if there are no any sockets with activities.
    */
    //This function checks to see if data is available for reading on the
    //given set of sockets.  If 'timeout' is 0, it performs a quick poll,
    //otherwise the function returns when either data is available for
    //reading, or the timeout in milliseconds has elapsed, which ever occurs
    //first.  This function returns true if there are any sockets ready for reading,
    //or false if there was an error with the select() system call.
    bool checkSockets(agx::UInt32 timeoutMillis);

    const SocketRefVector& getSockets() const { return m_set; }

  protected:

    /**
    \brief Destroys the socket set.
    Note, that it does not destroy the sockets this set holds references to.
    */
    virtual ~SocketSet(){
      m_set.clear();
    }

  private:

    SocketRefVector m_set;

  };
}
