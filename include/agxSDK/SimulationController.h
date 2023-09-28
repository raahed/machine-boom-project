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


#ifndef AGX_SIMULATION_CONTROLLER_H
#define AGX_SIMULATION_CONTROLLER_H

#include <agx/config/AGX_USE_WEBSOCKETS.h>

#if AGX_USE_WEBSOCKETS()

#include <agx/Referenced.h>
#include <agx/Callback.h>
#include <agx/Event.h>
#include <agx/macros.h>
#include <agx/Journal.h>
#include <agx/JournalReader.h>

#include <agxIO/RemoteCommandServer.h>

#include <agxSDK/Simulation.h>
#include <agxSDK/CallbackWrappers.h>

#include <agxNet/WebSocket.h>

namespace agx
{
  class Journal;
  typedef agx::ref_ptr<Journal> JournalRef;
}


namespace agxData
{
  class Track;
  typedef agx::ref_ptr<Track> TrackRef;
}



namespace agxSDK
{
  class RunLoopThread;
  class ScopedStepLock;


  /**
   * The Application class is a layer between an application and an agxSDK simulation. It
   * facilitates the transfer of state changes between the simulation and the application
   * using it.
   */
  AGX_DECLARE_POINTER_TYPES(SimulationController);
  class CALLABLE AGXPHYSICS_EXPORT SimulationController : public agx::Referenced
  {
  public:

    /**
     * Create an application with the RemoteCommandServer configuration found in the settings file.
     */
    SimulationController();


    /**
     * \param enableRemoteCommandServer If true, then a RemoteCommandServer is started and control callbacks bound.
     * \param port The port number that should be used if a RemoteCommandServer is to be started. Pass agx::InvalidIndex to use the value found in the settings.
     * \param scan If true, then the RemoteCommandServer will search for a available port and may use another port than the given one.
     */
    SimulationController( bool enableRemoteCommandServer, agx::UInt16 port = agx::InvalidIndex, bool scan = false );


    agxSDK::Simulation* getSimulation();
    const agxSDK::Simulation* getSimulation() const;

    agx::Journal* getJournal();
    const agx::Journal* getJournal() const;


    /**
     * Replace the current simulation with a new, empty one. The current journal settings
     * will also be lost.
     */
    void clearSimulation();


    /**
     * Enter a run loop that will repeatedly call 'step' until 'stopRunLoop' is called.
     * The thread can be controlled using the 'play' and 'pause' methods.
     *
     * There should not be multiple threads executing in 'runLoop' simultaneously.
     */
    void runLoop();


    /**
     * Set a flag that will cause the thread running 'runLoop' to eventually return;
     */
    void stopRunLoop();


    /**
     * Spawn a new thread that calls 'runLoop'.
     */
    void startRunLoopThread();



    /**
     * Cause the thread waiting in 'runLoop' to start performing simulation steps.
     */
    void play(bool wasTemporarilyPaused = false);


    /**
     * If currently continuously stepping in 'runLoop', then stop and wait after
     * the current step is completed. Stepping is resumed again using 'play'.
     */
    void pause(bool isTemporarilyPaused = false);


    /**
     * Step until the specified time is reached.
     */
    void stepTo(agx::Real time, CallbackWrapper_String* errorCallback = nullptr, bool wasTemporarilyPaused = false);

    /**
     * Step until the specified frame is reached.
     */
    void stepToFrame(agx::UInt frameIndex, CallbackWrapper_String* errorCallback = nullptr, bool wasTemporarilyPaused = false);

    /**
     * When paused, causes the run loop to take a single simulation step. Does nothing
     * if currently in play mode.
     */
    void singleStep( bool asynchronous = true );

    /**
     * When repeat mode is enabled, the playback will restart from frame 0 when reaching the end.
     */
    void setRepeat( bool repeat );
    bool getRepeat() const;

    /**
     * When real-time mode is enabled, then the 'runLoop' can hold off a call to
     * 'Simulation::step' in order to not get ahead of itself.
     */
    void setRealTimeSync( bool realTime );
    bool getRealTimeSync() const;

    /**
     * This can be used to skip frames in playback if the time step is really small.
     */
    void setFrameStride( agx::UInt stride );

//     /**
//      * Space is updated when a new frame is loaded if update space is enabled.
//      * It can be used to get updated contacts without saving them to the journal,
//      * but it might be expensive.
//      */
//     void setUpdateSpace( bool updateSpace );
//     bool getUpdateSpace() const;

    /**
     * Open the given journal. If there already is an open journal, then the older
     * one will be closed. If we were recording, then the recording will be stopped.
     *
     * If the second argument, 'sessionName', is non-empty, then the SimulationController
     * will enter playback mode and read from that session. WILL NOT RELOAD THE SCENE.
     * It is the responsibility of the caller to ensure that the loaded session is
     * compatible with the current content of the simulation.
     *
     * If 'sessionName' is empty, then the SimulationController will be ready to record
     * from the running simulation into the default session in the newly opened journal.
     */
    bool openJournal( const agx::String& journalPath="Journal.agxJournal", const agx::String& sessionName="", bool loadScene=true, bool attach=true );

    void setJournalFormat(agx::Journal::ArchiveFormat format);
    agx::Journal::ArchiveFormat getJournalFormat() const;

    bool sanityCheck() const;

    void attach();

    void setDefaultSceneLoader(agx::Journal::SceneLoaderCallback callback);

    bool recordExtraData( const agx::String& key, const agx::String& value );
    agx::String retrieveExtraData( const agx::String& key );

    bool truncate(agx::UInt firstFrameToRemove);

    /**
    Specifies if the simulation serialization should be temporary stored to disk or to memory when creating a journal.
    Storing to memory is default. But this might cause problem, as a large continuous memory need to be allocated.
    For 32bit applications, it might be problematic to allocate 100Mb or more continuous memory, hence storing to disk might be better. And slower.

    DEFAULT: store to memory (false)

    \param flag - if true, scene will be stored to disk, otherwise to memory before written to journal.
    */
    void setSaveSceneToDisk( bool flag );

    /**
    \return true if simulation scene will be temporarily stored to disk when writing a journal.
    */
    bool getSaveSceneToDisk() const;


    /**
     * Signal to the plot that the scene has changed (the plot is now invalid).
     */
    void signalSceneChanged();

    bool checkAndResetPlotSavedState();

    /**
    Enable/disable the listener that collects all statistics data from the simulation. Disabling will remove/delete
    the previous listener, hence no data will be stored
    */
    void setEnableSimulationStatisticsListener( bool flag );

    bool getEnableSimulationStatisticsListener( ) const;

    /**
     * Fills the given StringVector with the names of the currently available sessions.
     */
    bool listSessions( agx::Vector<agx::Name>& sessionNames );


    /**
     * Returns the name of the current session. Will be the empty string if there is no session.
     */
    agx::String getCurrentSessionName();


    bool deleteSessionFromDisk( const agx::String& sessionName );

    bool startRecording( const agx::String& journalConfiguration, const agx::String& journalPath="Journal.agxJournal", const agx::String& sessionName="" );
    bool resumeRecording( const agx::Name& sessionName, const agx::String& journalConfiguration );
    bool stopRecording(bool resetScene=true, bool blocking=false);
    bool startTrack();
    bool stopTrack();
    bool isTrackRunning() const;
    bool isRecording() const;
    bool isRunning() const;

    /**
     * Rename a recording while the track is running is NOT safe.
     * stopTrack can be used to be able to rename, but the attach method needs to be called after.
     */
    bool renameRecording( const agx::String& sessionName );

    /**
      * Set the frequency with which the journal records the data
    */
    agx::Real setJournalFrequency( agx::Real freq );

    /**
      * Get the frequency with which the journal records the data.
      * -1.0 if no journal exists.
    */
    agx::Real getJournalFrequency() const;

    /**
      * Get the stride with which journal records the data.
      * agx::UInt(-1) if no journal exists.
    */
    agx::UInt getJournalStride() const;

    agx::UInt getAccumulatedNumFrames() const;
    agx::UInt getNumQueuedFrames() const;
    agx::UInt getNumCachedFrames() const;

    /**
     * Copy the given session from the active journal into the journal with the given name. The other
     * journal will be created if it does not yet exist. The transfer will fail if the target journal
     * already contains a session with the same name.
     *
     * The journal at newJournalPath must not be open by another SimulationController, or other journal
     * inspecting/modifying classes.
     */
    bool copyRecording( const agx::String& sessionName, const agx::String& newJournalPath );

    /**
     * Move the given session from the active journal into the journal with the given name. The other
     * journal will be created if it does not yet exist. The transfer will fail if the target journal
     * already contains a session with the same name.
     *
     * The journal at newJournalPath must not be open by another SimulationController, or other journal
     * inspecting/modifying classes.
     */
    bool transferRecording( const agx::String& sessionName, const agx::String& newJournalPath );

    bool jumpToFrame( agx::UInt frame );
    bool jumpToTime( agx::Real time );

    agx::Real getTime() const;
    agx::UInt getCurrentFrameIndex() const;
    agx::Real getFirstFrameTimeStamp() const;
    agx::Real getLastFrameTimeStamp() const;
    agx::UInt getFirstFrameIndex() const;
    agx::UInt getLastFrameIndex() const;
    agx::UInt getNumFrames() const;
    agx::UInt16 getPort() const;


    /**
     * End playback from journal and optionally restore the simulation state to the start of the
     * recording. Further stepping will generate new simulation data.
     *
     * \param resetScene When true, then the simulation state will be restored to the start of the recording.
     *                   When false, then the current state of the simulation is kept for further simulation.
     */
    bool endPlayback( bool resetScene = true);

    void registerStartPlaybackSignalCallback(CallbackWrapper_Real* callback);
    void registerStartRecordingSignalCallback(CallbackWrapper_Real* callback);
    void registerStopPlaybackSignalCallback(CallbackWrapper_Real* callback);
    void registerStopRecordingSignalCallback(CallbackWrapper_Real* callback);

    void registerPreStepSignalCallback( CallbackWrapper_Real* callback );
    void registerPostStepSignalCallback(CallbackWrapper_Real* callback);
    void registerPlaySignalCallback( CallbackWrapper* callback );
    void registerPauseSignalCallback( CallbackWrapper* callback );
    void registerRealTimeSignalCallback( CallbackWrapper_Bool* callback );
    void registerJumpSignalCallback( CallbackWrapper_Real* callback );
    void registerErrorSignalCallback( CallbackWrapper_String* callback );


    void unregisterErrorSignalCallback( CallbackWrapper_String* callback );
    void unregisterPreStepSignalCallback( CallbackWrapper_Real* callback );
    void unregisterPostStepSignalCallback(CallbackWrapper_Real* callback);

    void unregisterStartPlaybackSignalCallback(CallbackWrapper_Real* callback);
    void unregisterStartRecordingSignalCallback(CallbackWrapper_Real* callback);
    void unregisterStopPlaybackSignalCallback(CallbackWrapper_Real* callback);
    void unregisterStopRecordingSignalCallback(CallbackWrapper_Real* callback);

    /**
     * The method is used to gain exclusive access to the SimulationController and the
     * simulation it holds. When 'requestExclusiveAccess' has returned, then the run loop
     * won't perform any stepping and all of the manipulator methods (from other threads)
     * will block until 'relinquishExclusiveAccess' has been called.
     *
     * There is a helper class that can be used to avoid deadlock bugs due to forgetting
     * to relinquish the exclusive access somewhere. This class is called ScopedStepLock
     * and is used as follows, where 'simulationController' is a pointer to an instance of
     * the agxSDK::SimulationController class.
     *
     * \code
     * {
     *   axSDK::ScopedStepLock stepLock( simulationController );
     *   // We now have exclusive access to the 'simulationController' object.
     * }
     * // The 'stepLock' variable just went out of scope and relinquished the exclusive access.
     * \endcode
     */
    void requestExclusiveAccess();
    void relinquishExclusiveAccess();

    void setupRemoteCommandServer( agx::UInt16 port = agx::InvalidIndex, bool scan = false );

    /**
    Get all the contact (points and normals) from the buffer in the current Simulation.
    If no contacts are found the vectors \p contactPoints and \p normals are cleared
    \return false if zero contacts are available, otherwise true
    */
    bool getContacts( std::vector<agx::Vec3>& contactPoints, std::vector<agx::Vec3>& normals, std::vector<agx::Real>& depths ) const;

    /**
    This method will call Space::updated() get all the contacts, generate contact materials, and calculate the number
    of contacts using the direct solver. This can then be used to estimate the complexity of the system.

    Methods called:

    Space::update()
    MaterialManager::updateContactMaterials()
    */
    size_t calculateDirectSolvedContacts() const;

  protected:
    virtual ~SimulationController();


  private:
    void stepTo(std::function<bool(agx::Clock *)> stopPredicate, CallbackWrapper_String* errorCallback, bool wasTemporarilyPaused);

    // 'step', as it's written now with synchronous execution, doesn't fit well into the threading model used for the continuous stepping.
    // Need a way to make the 'runLoop' thread we woken and take a single step. Do we need a public API 'step' method?
    /**
     * Step the simulation once. Called from 'runLoop'.
     */
    void step();

    /*
     * HTTP callbacks used by the RemoteCommandServer.
     */
    // Basic simulation control.
    bool httpPlaySimulation(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpPauseSimulation(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpEnableRealTimeSync(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpDisableRealTimeSync(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    // Journal control.
    bool httpEnableRecording(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpDisableRecording(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpListJournals(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpLoadJournal(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpUnloadJournal(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpSetPlaybackPosition(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    // RemoteViewer control.
    bool httpAttachRemoteViewer(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);


    // Websocket setup.
    bool httpGetControlChannel(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);
    bool httpInitSynchronizationSocket(agxIO::RemoteCommandServer *server, mg_connection *connection, const agxIO::RemoteCommandServer::QueryArgumentTable& arguments);

    void signalStartPlayback();
    void signalStartRecording();
    void signalStopPlayback();
    void signalStopRecording();

    // Signaling methods. This will trigger the callbacks and write messages to the socket clients.
    void signalPreStep();
    void signalPostStep();
    void signalPlay();
    void signalPause();
    void signalRealTime();
    void signalJump();
    void signalError(agx::String error);


  private:
    void init();
    void webSocketClientConnected( agxNet::WebSocket* client );
    void webSocketClientDisconnected( agxNet::WebSocket* client );
    void webSocketMessageReceived( agxNet::WebSocket *socket, agx::UInt8* payload, size_t size );
    void sendWebSocketMessage(const agxJson::Value& eMessage);


  private:
    agx::Journal::ArchiveFormat m_journalFormat;

    bool m_realTime;

    /** Flag used to tell the 'runLoop' thread that it should end the run loop. */
    bool m_exitRunLoop;

    /** Flag used to let the run loop thread know that it should take a single step. */
    // bool m_singleStep;

    /** Atomic counter to keep track of how many single steps we have left to take. This is
     * required since 'singleStep' is non-blocking and multiple calls to 'singleStep' should
     * result in the same number of actual steps taken. */
    agx::AtomicValue m_singleStepsToTake;


    agxSDK::SimulationRef m_simulation;
    agx::JournalRef m_journal;



//    agx::RemoteCommandServer m_commandServer; ///\todo Make the RemoteCommandServer not a singleton.

    // These events are called from the simulation thread.
    agx::Event1<agx::Real> m_startPlaybackEvent;
    agx::Event1<agx::Real> m_startRecordingEvent;
    agx::Event1<agx::Real> m_stopPlaybackEvent;
    agx::Event1<agx::Real> m_stopRecordingEvent;

    // These events might be called for the main gui thread.
    agx::Event1<agx::Real> m_preStepEvent;
    agx::Event1<agx::Real> m_postStepEvent;
    agx::Event m_playEvent;
    agx::Event m_pauseEvent;
    agx::Event1<bool> m_realTimeEvent;
    agx::Event1<agx::Real> m_jumpEvent;
    agx::Event1<agx::String> m_errorEvent;

    agx::Vector< agx::ref_ptr<CallbackWrapperBase> > m_callbacks;

    agxNet::WebSocketRefVector m_webSocketClients;
    agxNet::WebSocketServerRef m_webSocketServer;

    agx::Journal::SceneLoaderCallback m_defaultSceneLoader;

    agxSDK::RunLoopThread* m_runLoopThread;

    /** A mutex that must be taken before the simulation is manipulated. Held by the
     * run loop thread for the duration of every 'Simulation::stepForward' call. */
    agx::ReentrantMutex m_frameMutex;



    /** Used to temporarily prevent the run loop thread from re-taking the frame mutex
     * at the transition between two consecutive time steps. Is intended to be used by
     * other ( e.g. the HTTP server ) threads in a reset-lock-release-unlock cycle together
     * with the frame mutex in order to prevent thread starvation. The 'runLoop' thread
     * is pretty quick to re-take the lock after it's been released.
     */
    agx::Block m_frameBlock;

    /** This block will be blocking when the simulation is paused, causing the run loop
     * thread to block on it.
     */
    agx::Block m_pauseBlock;

    /** This block will be blocking while the simulation is stopped and the stop callback is not called yet.
     */
    agx::Block m_stopBlock;

    /** Flag that should always mirror the state of the 'm_pauseBlock' thread synchronization primitive. */
    bool m_autoStep;

    /**
     * True while we are recording, i.e, when the playback mode is FROM_SIMULATION
     * and the journal is attached to the simulation in Journal::RECORD mode.
     *
     * Can never be true while playback mode is FROM_JOURNAL.
     */
    bool m_isRecording;

    /**
     * Shows from which source the simulation is driven. FROM_SIMULATION means that the
     * simulation is generating new data, using the solver and all that. FROM_JOURNAL
     * means that we have an open journal that frames are being read from.
     *
     * Can never be FROM_JOURNAL while 'm_isRecording' is true.
     */
    enum PlaybackMode {FROM_SIMULATION, FROM_JOURNAL} m_playbackMode;


    /**
     * This is the track that is used to communicate simulation data with remote hosts
     * such as the plotting tool. Not created until first asked for.
     */
    agxData::TrackRef m_remoteViewerTrack;

    /**
     * The server port to which remote viewer clients can connect.
     */
    agx::UInt16 m_remoteViewerPort;

    /**
     * This determines if Space should be updated when a new frame is loaded.
     * It can be used to get updated contacts without saving them to the journal,
     * but it might be expensive.
     */
    // bool m_updateSpace;


    // This block of members are used for communication with the plotting tool.

    agxNet::WebSocket::ControlChannelRef m_controlChannel;
    void controlChannelSocketConnectedCallback( agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket );
    agxNet::WebSocket::ControlChannel::SocketEvent::CallbackType m_controlChannelSocketConnectedCallback;
    agxNet::WebSocket::ControlChannel::SocketEvent::CallbackType m_controlChannelSocketDisconnectedCallback;

    void ccGetTreeStructure( agxNet::WebSocket::ControlChannel* channel, agxNet::WebSocket* socket, agxNet::StructuredMessage* message );
    agxJson::Value& createArrayElement( agxJson::Value& eParent, const char* groupName );
    void signalPlaybackStarting();
    void signalPlaybackEnding();

    agx::JournalReaderRef m_journalReader;

    bool m_simulationStatisticsListenerEnabled;
    bool m_saveSimulationToDisk;

    bool m_shouldSignalStartPlayback;
    bool m_shouldSignalStartRecording;
    bool m_shouldSignalStopPlayback;
    bool m_shouldSignalStopRecording;
    bool m_shouldJumpAfterStop;
    agx::UInt m_jumpAfterStopFrame;
    bool m_isTemporarilyPaused;

  };


  /**
   * Helper class that acquires and later relinquishes the step lock of a SimulationController.
   */
  class AGXPHYSICS_EXPORT ScopedStepLock
  {
  public:
    ScopedStepLock( SimulationController* simulationController );
    ~ScopedStepLock();
  private:
    ScopedStepLock( const ScopedStepLock& ) {};
    void operator=( const ScopedStepLock& ) {};

    SimulationController* m_simulationController;
  };





  //// Implementation.
  AGX_FORCE_INLINE agxSDK::Simulation* SimulationController::getSimulation() { return m_simulation; }
  AGX_FORCE_INLINE const agxSDK::Simulation* SimulationController::getSimulation() const { return m_simulation; }
  AGX_FORCE_INLINE bool SimulationController::getRealTimeSync() const { return m_realTime; }
  //AGX_FORCE_INLINE bool SimulationController::getUpdateSpace() const { return m_updateSpace; }
  AGX_FORCE_INLINE bool SimulationController::isRecording() const { return m_isRecording; }
  AGX_FORCE_INLINE bool SimulationController::isRunning() const { return m_autoStep; }

  AGX_FORCE_INLINE const agx::Journal* SimulationController::getJournal() const { return m_journal; }
  AGX_FORCE_INLINE agx::Journal* SimulationController::getJournal() { return m_journal; }
}

// use websockets?
#endif

#endif
