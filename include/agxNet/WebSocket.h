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

#include <agx/Callback.h>
#include <agx/HashTable.h>
#include <agx/Referenced.h>
#include <agxData/Frame.h>
#include <agxData/BinaryData.h>
#include <agxNet/FramePacket.h>
#include <agxNet/PortRange.h>
#include <external/json/value.h>
#include <queue>



// Forward declarations of libwebsockets stuff so we can pass around opaque pointers.
struct lws_context;
struct lws_protocols;
struct lws;


namespace agxNet
{
  class WebSocketServer;
  AGX_DECLARE_POINTER_TYPES(WebSocket);
  AGX_DECLARE_VECTOR_TYPES(WebSocket);

  /**
  A socket that connects to a client. Instances of this class is created by the
  Socket server when a new client connects.
  */
  class AGXCORE_EXPORT WebSocket : public agx::Referenced
  {
  public:

    AGX_DECLARE_POINTER_TYPES(Packet);
    AGX_DECLARE_VECTOR_TYPES(Packet);
    AGX_DECLARE_POINTER_TYPES(ControlChannel);



    /**
     * A callback function that is run when a message is received from the socket.
     * The first argument is the WebSocket that received them message, the second
     * is a pointer pointing to the received message and the third argument is the
     * size of the message in bytes.
     */
    typedef agx::Callback3<WebSocket *, agx::UInt8*, size_t> ReceiveCallback;


    /**
     * Set the callback function that should be called when new messages are received.
     */
    void setReceiveCallback(ReceiveCallback callback);


    /**
     * Send the content of the given packet to the client.
     */
    void send(const agx::String& message);
    void send(Packet* packet);
    void send(const StructuredMessage *message);


    // For recieving partial packages
    void resetPartialMessage();
    bool hasPartialMessage();
    void appendPartialMessage(void* pmsg, size_t size, size_t remaining);
    void receivePartialMessage();


    // For libwebsockets only.
    void messageReceived(void* message, size_t size);

    void setAnonymous(bool flag);
    bool isAnonymous() const;

    bool isAuthenticated() const;
    void setAuthenticated(bool flag);

  protected:
    virtual ~WebSocket();

  private:
    friend class WebSocketServer;

    /**
     * WebSockets should not be created directly. This is done by WebSocketServer.
     */
    WebSocket(WebSocketServer* context, lws* socket);

  private:
    WebSocketServer* m_context;
    lws* m_wsiHandle;
    ReceiveCallback m_receiveCallback;
    agx::UInt32 m_nextMessageId;
    bool m_anonymous;
    bool m_authenticated;

    // For partial message handling
    agx::UInt8* m_partialMessage;
    size_t m_partialMessageCurrOffset;

    struct PacketTransfer
    {
      PacketTransfer(Packet *p) : packet(p), numWritten(0) {}
      PacketTransfer() : packet(nullptr), numWritten(0) {}

      PacketRef packet;
      size_t numWritten;
    };

    typedef std::queue<PacketTransfer> PacketQueue;
    PacketQueue m_sendQueue;
  };


  /**
  WebSocket::Packet
  */
  class AGXCORE_EXPORT WebSocket::Packet : public agx::Referenced
  {
  public:
    enum ContentType
    {
      CONTENT_BINARY,
      CONTENT_TEXT
    };

    Packet(size_t size, ContentType type);

    // Convenience method to create a text packet
    Packet(const agx::String& message);
    Packet(const agxJson::Value& message);

    // Convenience method to create a data frame packet
    Packet(const agxData::Frame *frame);


    /**
     * Returns a pointer to the data this Packet contains. Senders should write their
     * message to this memory area.
     */
    agx::UInt8* getPayload();


    /*
     * Returns the size of the payload memory area, in bytes.
     */
    size_t getSize() const;


    ContentType getContentType() const;




    // Internal
    int getLibwebsocketsWriteProtocol() const;

  private:
    friend class WebSocket;
    void init(size_t size, ContentType type);

  protected:
    virtual ~Packet();

  private:
    ContentType m_type;

    // We use agx::UInt8* because of weaker pointer aliasing rules.
    agx::UInt8* m_buffer;   // Includes both metadata and the payload itself.
    agx::UInt8* m_message;  // Pointer into the m_buffer memory segment, at the start of the payload.
    size_t m_messageSize;
  };


  /**
  WebSocketServer
  */
  AGX_DECLARE_POINTER_TYPES(WebSocketServer);
  class AGXCORE_EXPORT WebSocketServer : public agx::Referenced
  {
  public:

    /**
     * A callback function that is called whenever a new client connects. The argument
     * is the agxNet::WebSocket instance created for the new client.
     */
    typedef agx::Callback1<WebSocket*> SocketCallback;


    WebSocketServer(agx::UInt16 port, SocketCallback connectedCallback = SocketCallback(), SocketCallback disconnectedCallback = SocketCallback(), const agx::String& protocol = "agx_stream");
    WebSocketServer(agxNet::PortRange *range, SocketCallback connectedCallback = SocketCallback(), SocketCallback disconnectedCallback = SocketCallback(), const agx::String& protocol = "agx_stream");


    /**
     libwebsocket doesn't do any threading by itself, so event servicing must be triggered
     manually. Returns immediately if there are no new events.

     \return true if activity on the socket
     */
    bool service(agx::Real timeoutMs = 0);

    /**
    \return The port the server is listening on.
    */
    agx::UInt16 getPort() const;

    /**
    \return True if using SSL
    */
    bool useSSL() const;


    WebSocket *createConnection(const agx::String& address, agx::UInt16 port, bool useSSL = false, const agx::String& path = "/", const agx::String& protocol = "agx_stream", bool isAnonymous = false);

    void send(WebSocket::Packet* packet);
    void send(const StructuredMessage* message);

    /*
     * The following is for libwebsockets only. Should not be called by users.
     */

    static WebSocketServer* getServer(lws_context* contextHandle);
    WebSocket *acceptNewClient(lws* clientHandle);
    WebSocket* getWebSocket(lws* clientHandle);
    void connectionClosed(lws* clientHandle);
    void connectionDestroyed(lws* clientHandle);

    class SocketEventHandler;

  public:
    typedef agx::HashTable<lws*, WebSocketRef> SocketTable;
    const SocketTable& getClientSockets();

  protected:
    virtual ~WebSocketServer();
    void initProtocols(const char *protocolName);
    void init(agx::UInt16 port);

  private:
    friend class WebSocket;
    agx::UInt16 m_port;
    bool m_runningService;
    bool m_hasNewSocketEvent;

    lws_context* m_contextHandle;
    SocketCallback m_connectedCallback;
    SocketCallback m_disconnectedCallback;

    SocketTable m_sockets;

    typedef agx::HashTable<lws_context*, WebSocketServer*> ContextTable;
    static ContextTable s_contexts;
    lws *m_currentCreatedConnection;
    agx::String m_protocolName;
    lws_protocols *m_protocols;
    bool m_useSSL;
  };


  /**
  ControlChannel
  */
  class AGXCORE_EXPORT WebSocket::ControlChannel : public agx::Referenced
  {
  public:
    typedef agx::Event2<ControlChannel *, agxNet::WebSocket *> SocketEvent;

    SocketEvent socketConnectEvent;
    SocketEvent socketDisconnectEvent;

  public:
    ControlChannel(agx::UInt16 port, const agx::String& token = "");
    ControlChannel(agxNet::PortRange* portRange, const agx::String& token = "");

    agx::UInt16 getPort() const;

    agxNet::WebSocket *createConnection(const agx::String& address, agx::UInt16 port, bool useSSL = false, const agx::String& path = "/", const agx::String& protocol = "agx_stream", bool isAnonymous = false);
    void send(const StructuredMessage *message);

    bool service(agx::Real timeoutMs = 0);

    typedef agx::Callback3<ControlChannel *, agxNet::WebSocket *, agxNet::StructuredMessage *> MessageHandler;

    void registerMessageHandler(const agx::String& uri, MessageHandler callback);

  protected:
    virtual ~ControlChannel();

  private:
    void clientConnected(agxNet::WebSocket* client);
    void clientDisconnected(agxNet::WebSocket* client);
    void messageReceived(agxNet::WebSocket *socket, agx::UInt8* payload, size_t size);

  private:
    agxNet::WebSocketServerRef m_serverSocket;
    agxNet::WebSocketRefVector m_clientSockets;

    typedef agx::HashTable<agx::String, MessageHandler> MessageHandlerTable;
    MessageHandlerTable m_messageHandlerTable;
    agx::String m_token;
    agxNet::StructuredMessageRef m_currentMessage;
    agxNet::WebSocket *m_currentMessageSocket;
  };

  /////////// Implementation
  AGX_FORCE_INLINE agx::UInt8* WebSocket::Packet::getPayload() { return m_message; }
  AGX_FORCE_INLINE size_t WebSocket::Packet::getSize() const { return m_messageSize; }
  AGX_FORCE_INLINE WebSocket::Packet::ContentType WebSocket::Packet::getContentType() const { return m_type; }


  AGX_FORCE_INLINE agx::UInt16 WebSocketServer::getPort() const { return m_port; }

}

// AGX_USE_WEBSOCKETS
#endif

