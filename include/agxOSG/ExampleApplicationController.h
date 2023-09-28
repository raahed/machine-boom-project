/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXOSG_EXAMPLEAPPLICATIONCONTROLLER_H
#define AGXOSG_EXAMPLEAPPLICATIONCONTROLLER_H

#include <agx/config/AGX_USE_WEBSOCKETS.h>
#include <agx/config/AGX_USE_WEBPLOT.h>

#include <agxOSG/PickHandler.h>
#include <agxOSG/VideoStream.h>
#include <agx/Component.h>
#include <agx/Journal.h>
#include <agx/JournalReader.h>
#include <agxNet/Socket.h>
#include <agxNet/WebSocket.h>
#include <agxNet/PortRange.h>
#include <agxIO/RemoteCommandServer.h>
#include <agxSDK/PickHandler.h>

#include <agxPlot/System.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Geometry>
#include <osg/PositionAttitudeTransform>
#include <osg/LineWidth>
#include <osgViewer/Viewer>
#include <osg/ShapeDrawable>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxData/BinaryData.h>

#include <agx/Physics/Geometry/TrimeshEntity.h>
#include <agxCollide/Trimesh.h>
#include <queue>
#include <agxCallable/CallableAccessor.h>

#if AGX_USE_WEBSOCKETS()


namespace agxOSG
{
  class ExampleApplication;

  AGX_DECLARE_POINTER_TYPES(ExampleApplicationController);
  class AGXOSG_EXPORT ExampleApplicationController : public agx::Component
  {
  private:
    AGX_DECLARE_POINTER_TYPES(ControlChannelTCP);

  public:
    ExampleApplicationController(ExampleApplication *application);

    void signalAutoStepping(bool flag);
    void pushParameters();
    void pushFrame();
    void setEnableTickSignals(bool flag);
    bool update(agx::Real timeoutMs = 0);
    void signalRefresh();
    void updateMouseSpringRenderer();

    void signalJournalPlayback(agx::Journal *journal);
    void signalJournalRecord(agx::Journal *journal);

    /// \todo This is part of the HTTP/OpenGL threads hack.
    bool needSceneLoading();
    void performSceneLoading();

    struct RenderFrame;
    void updateRemoteViewers(const RenderFrame& frame);

    void setOrbitCamera(agxCollide::Geometry* geometry, const agx::Vec3& eye, const agx::Vec3& center, const agx::Vec3& up, int trackerMode);

    void init(const agx::String& token = "", agxNet::PortRangeRef portRange = nullptr);

#if AGX_USE_WEBPLOT()
    void initPlotCallbacks();
#endif

    void initDjangoSocket(const agx::String& address, const agx::String& token);
    void initRemoteConnection();

    agxNet::WebSocket::ControlChannel *getControlChannel();

  protected:
    virtual ~ExampleApplicationController();

  private:
    bool httpGetCurrentState(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpStartSimulation(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpStopSimulation(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpStepForward(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpStepBackward(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpListLuaScenes(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpLoadScene(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpListJournals(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpAddJournalBinding(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpRemoveJournalBinding(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpListJournalBindings(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpSaveCurrentJournalBindings(agxIO::RemoteCommandServer* server, mg_connection* connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpEnableRecording(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpDisableRecording(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpLoadJournal(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpUnloadJournal(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);


#if AGX_USE_WEBSOCKETS()
    void webSocketClientConnected(agxNet::WebSocket* client);
    void webSocketClientDisconnected(agxNet::WebSocket* client);
#endif

    void simulationTickCallback(agx::Clock *clock);


    bool httpEnablePlaybackLoop(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpDisablePlaybackLoop(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpEnableRealTimeSync(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpDisableRealTimeSync(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpSetPlaybackPosition(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpAttachVideoStream(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
//    bool httpAttachRemoteViewer(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
//    bool httpAttachRemoteViewerSlave(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpAttachRemoteSolver(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpDetachRemoteSolver(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpAttachStatisticsViewer(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpGetControlChannel(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    bool httpGetRemoteViewChannel(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    void buildStateMessage(agxJson::Value& eRoot);

    #if AGX_USE_WEBSOCKETS()
    void ccGetScene(ControlChannelTCP *channel, agxNet::TCPSocket *socket, agxNet::StructuredMessage *message);
    void ccSetScene(ControlChannelTCP *channel, agxNet::TCPSocket *socket, agxNet::StructuredMessage *message);
    void ccAttachRemoteViewer(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccRegisterUser(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccRegisterAttributeListener(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccUnregisterAttributeListener(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccRegisterCallableDataStream(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccFunctionCall(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccHeartbeat(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccEnableStructureSynchronization(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket, agxNet::StructuredMessage *message);
    void ccGetTreeStructure(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccGetTrimeshData(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccGetShapeRenderData(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccGetState(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccExitApplication(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccReloadScene(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccLoadSceneCallback(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message, std::function<void ()> callback);
    void onSceneLoad(agxNet::WebSocket* socket, agxNet::StructuredMessage* message, agxNet::StructuredMessage* response,
std::function<void ()> callback);
    void ccLoadScene(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccKeyDown(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccKeyUp(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccMousePickStart(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccMousePickDrag(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccMousePickEnd(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccPlay(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);
    void ccPause(agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message);

    void registerPort(agx::UInt16 port, const agx::String& portName);
    void unregisterPort(agx::UInt16 port, const agx::String& portName);
    void djangoMessage(agxNet::WebSocket *, agx::UInt8* data, size_t numBytes);


    void getConstraintData(agx::Constraint *constraint, agxJson::Value& eConstraint);
    void getRigidBodyData(agx::RigidBody *body, agxJson::Value& eBody);

    agxJson::Value& createArrayElement( agxJson::Value& eParent, const char* groupName );

    AGX_DECLARE_POINTER_TYPES(StructureSynchronization);
    class StructureSynchronization : public agx::Referenced
    {
    public:
      StructureSynchronization(agx::Component *root);

      void addSocket(agxNet::WebSocket *socket);

    protected:
      virtual ~StructureSynchronization();

    private:
      void addCallback(agx::Component *parent, agx::Object *child);
      void removeCallback(agx::Component *parent, agx::Object *child);

      agxNet::StructuredMessage *buildMessage(agx::Component *root);

    private:
      agx::ComponentObserver m_root;
      agx::Component::ObjectEvent::CallbackType m_addCallback;
      agx::Component::ObjectEvent::CallbackType m_removeCallback;
      agxNet::WebSocketRefVector m_sockets;
    };

    typedef agx::HashTable<agx::Component *, StructureSynchronizationRef> StructureSynchronizationTable;
    StructureSynchronizationTable m_structureSynchronizationTable;
    #endif

  private:
    AGX_DECLARE_POINTER_TYPES(MouseSpringRenderer);
    class MouseSpringRenderer : public agxData::FrameWriter
    {
    public:
      MouseSpringRenderer(ExampleApplication *application);

      void frameCallback(agxData::Frame *frame);
      virtual void writeFrame(const agxData::Frame *frame) override;
      virtual void endOfStream() override;

      // Blah, hack
      void update();

      agxData::Track::FrameEvent::CallbackType *getFrameCallback() { return &m_frameCallback; }
    protected:
      virtual ~MouseSpringRenderer();

    private:
      bool init();

    private:
      // osgViewer::Viewer *m_viewer;
      ExampleApplication *m_application;
      osg::PositionAttitudeTransform *m_mouseAttachTransform;
      osg::PositionAttitudeTransform *m_mousePositionTransform;
      osg::Geometry *m_mouseSpringLine;
      osg::ref_ptr< osg::Switch > m_mouseSpringSwitch;
      agxData::FrameConstRef m_frame;
      agxData::Track::FrameEvent::CallbackType m_frameCallback;
    };

    AGX_DECLARE_POINTER_TYPES(RemoteViewerThread);


  private:
    struct OrbitCamera
    {
      OrbitCamera() : enabled(false), mode(0) {}
      OrbitCamera(agxCollide::Geometry *g, agx::Vec3 e, agx::Vec3 c, agx::Vec3 u, int m) : enabled(true), mode(m), geometry(g), eye(e), center(c), up(u) {}

      bool enabled;
      int mode;
      agxCollide::Geometry *geometry;
      agx::Vec3 eye;
      agx::Vec3 center;
      agx::Vec3 up;
    };
    OrbitCamera m_orbitCamera;

    ExampleApplication *m_application;
    bool m_initialized;
    // agx::JournalRef m_journal;
    agx::Clock::TickEvent::CallbackType m_simulationTickCallback;
    bool m_enableTickCallback;
    MouseSpringRendererRef m_mouseSpringRenderer;
    agx::Timer m_rtTimer;
    agx::Real m_rtSmoothedRatio;
    agxData::TrackRef m_remoteViewerTrack;
    agx::UInt16 m_remoteViewerPort;
    bool m_remoteSimulationReceived;
    std::string loadSceneFormat;
    agx::Timer m_interactiveClientHeartbeatTimer;
    agx::Real m_interactiveClientHeartbeatTimeOut;

    RemoteViewerThreadRef m_remoteViewerThread;
    agxNet::PortRangeRef m_portRange;



    /// \todo Part of the HTTP/OpenGL threads hack.
    bool m_needSceneLoading;
    bool m_hasLooping;


    class ControlChannelTCP : public Referenced
    {
    public:
      ControlChannelTCP(ExampleApplicationController *controller, agx::UInt16 port);

      agx::UInt16 getPort() const;

      agxNet::TCPSocket *createConnection(const agx::String& address, agx::UInt16 port);

      bool service();

      typedef agx::Callback3<ControlChannelTCP *, agxNet::TCPSocket *, agxNet::StructuredMessage *> MessageHandler;

      void registerMessageHandler(const agx::String& uri, MessageHandler callback);

    protected:
      virtual ~ControlChannelTCP();

    private:
      void receiveMessage(agxNet::TCPSocket *socket, const agxNet::StructuredMessage::PreHeader& preHeader);

    private:
      agx::UInt16 m_port;
      agxNet::TCPServerSocketRef m_serverSocket;
      agxNet::TCPSocketRefVector m_clientSockets;

      typedef agx::HashTable<agx::String, MessageHandler> MessageHandlerTable;
      MessageHandlerTable m_messageHandlerTable;
    };


    bool m_setupComplete;

    #if AGX_USE_WEBSOCKETS()
    agxNet::WebSocket::ControlChannelRef m_controlChannel;
    agxNet::WebSocketRef m_djangoSocket;
    void controlChannelSocketConnectedCallback(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket);
    void controlChannelSocketDisconnectedCallback(agxNet::WebSocket::ControlChannel *channel, agxNet::WebSocket *socket);
    agxNet::WebSocket::ControlChannel::SocketEvent::CallbackType m_controlChannelSocketConnectedCallback;
    agxNet::WebSocket::ControlChannel::SocketEvent::CallbackType m_controlChannelSocketDisconnectedCallback;

#if AGX_USE_WEBPLOT()
    void plotSystemOutputAddedCallback(agxPlot::Output* output);
    agxPlot::System::OutputAddedEvent::CallbackType m_plotSystemOutputAddedCallback;
#endif

    struct UserData
    {
      agxSDK::PickHandlerRef pickHandler;
      agx::UInt64 id;
      agxJson::Value data;
      agx::Real pickDistance;
    };

    UserData& getUserData(agxNet::WebSocket *);

    typedef agx::HashTable<agxNet::WebSocket *, UserData> UserTable;
    UserTable m_userTable;

    typedef std::queue<agxNet::StructuredMessageRef> FrameMessageQueue;
    FrameMessageQueue m_frameMessageQueue;
    #endif

    ControlChannelTCPRef m_controlChannelTCP;
    agx::JournalReaderRef m_journalReader;

#if AGX_USE_FFMPEG()
    VideoStreamRef m_videoStream;
#endif

    agx::RigidBodyRefVector m_monitoredBodies;
    agx::ConstraintRefVector m_monitoredConstraints;
    bool m_djangoProxyResponse;
    agxCallable::CallableAccessorRefVector m_dataStreamCallables;
    agx::Callback m_sceneLoadCallback;
  };

  struct ExampleApplicationController::RenderFrame
  {
    osg::ref_ptr<osg::Image> image;
    agx::UInt index;
    agx::DateRef date;
  };

  class ExampleApplicationController::RemoteViewerThread : public agx::BasicThread, public agx::Referenced
  {
  public:
    RemoteViewerThread(ExampleApplication *application);

    void stop();

    void pushFrame(const RenderFrame& frame);

    agx::UInt16 getPort();

  protected:
    virtual ~RemoteViewerThread();

  private:
    virtual void run() override;

    void connectCallback(agxNet::WebSocket *client);
    void disconnectCallback(agxNet::WebSocket *client);
    void messageReceiveCallback(agxNet::WebSocket *, agx::UInt8 *data, size_t numBytes);
    void updatePickHandlers();

  private:
    agxNet::WebSocketServerRef m_serverSocket;
    bool m_running;
    agx::Condition m_condition;
    agx::Mutex m_mutex;
    RenderFrame m_currentFrame;
    ExampleApplication *m_application;

    typedef agx::HashTable<agxNet::WebSocket *, PickHandlerRef> PickHandlerTable;
    PickHandlerTable m_pickHandlerTable;

    agx::Vector< std::pair<PickHandlerRef, agxNet::StructuredMessageRef> > m_pickEvents;
  };

}

#endif

#endif /* AGXOSG_EXAMPLEAPPLICATIONCONTROLLER_H */
