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

#ifndef AGXREMOTESOLVERCLIENT_H
#define AGXREMOTESOLVERCLIENT_H


#include <agxRemoteSolverClient/export.h>
#include <agx/Referenced.h>
#include <agxNet/SocketTrack.h>


DOXYGEN_START_INTERNAL_BLOCK()

namespace agxNet
{
  class IPAddress;
}

namespace agxRemoteSolverClient
{

  /**
   * The purpose of RemoteSolverClient is to allow external applications to act as
   * solvers in the AGX simulation loop.
   *
   *
   * I'm prototyping with two forms of drivers: callbacks and read/write methods.
   * Eventually one of these will be chosen as the prefered way and the other
   * removed. If both are deemed valuable, then a class hierarchy with a
   * RemoteSolverClient base class and two derived classes should be considered.
   *
   * When using the callback version, the user supplies a callback function and
   * then calls the thread-stealing method 'beginSimulationLoop'. When in the
   * simulation loop, this class will wait for solver data from the AGX server.
   * When a simulation state is received, then RemoteSolverClient will call the
   * give 'onFrameReceived' callback. The return value of this function is a Frame
   * which should contain the result of solving the system, i.e, new velocities and
   * positions for the bodies and Lagrange multipliers for the constraints. The
   * system currently doesn't constraint which data can be send, so any buffer
   *  present on the server can be overwritten with data from the client application.
   *
   * The second mode of operation is through read/write functions. Whenever the
   * client application is ready to perform a new solve it calls 'readSystem' to get the
   * current simulation state. This call will block until the AGX server has supplied
   * a simulation state. When the received system has been solved the client application
   * calls the 'writeSolution' method, which will send the give Frame data back to
   * the server. Again, this data should include at least new velocities and positions
   * for the bodies and Lagrange multipliers for the constraints.
   *
   *
   * Note that in all cases, the Frame sent back to the server need not be the same
   * as the one received. If it is, then all unchanged buffers should be removed
   * prior to transmission in order to reduce bandwidth waste.
   */
  class /*AGXREMOTESOLVERCLIENT_EXPORT*/ RemoteSolverClient : public agx::Referenced  /// \TODO How should export declarations be setup in extra libraries?
  {
  public:

    /**
     * Function pointer typedef for the solve callback function. The solver can be
     * implemented in a function with a signature compatible with this typedef.
     */
    typedef agxData::Frame*(*FrameReceivedCallback)(const agxData::Frame* frame);

    /**
     * \@param server The address and port of the RemoteCommandServer. There is currently
     * no way to connect to a listening RemoteSolver directly.
     *
     * \@param onFrameReceived An optional callback function pointer.
     *
     * When 'onFrameReceived' is 0, then the RemoteSolverClient expects to receive
     * 'readSystem'/'writeSolution' calls every now and then. Users should not call
     * 'beginSimulationLoop' or 'cancel' in this case.
     *
     * When 'onFrameReceived' is not 0, then the RemoteSolverClient expects a single call to
     * 'beginSimulationLoop'. This will steal the executing thread and the given
     * 'onFrameReceived' function will be called whenever a system description has been
     * received from the server.
     */
    RemoteSolverClient( const agxNet::IPAddress& server, bool includeRemoteSolverSetup = true, FrameReceivedCallback onFrameReceived = 0 );


    enum ReadLoopEndReason{ BEEN_CANCELED, CONNECTION_CLOSED, CALLBACK_MISSING, CALLBACK_FAILED };

    /**
     * Only for use in callback mode.
     * Start the read/write loop, with calls to 'onFrameReceived' for each simulation
     * step done by the AGX server. Will not return until some time after a call to cancel
     * or when the AGX server closes the connection.
     */
    ReadLoopEndReason beginSimulationLoop();

    /**
     * Only for use in callback mode.
     * End the simulation loop, causing 'beginSimulationLoop' to return. Eventually. The
     * current TcdpSocketFrameReader implementation doesn't support timeouts, so
     */
    void cancel();


    /**
     * Only for use in the read/write mode.
     * Get the current simulation state from the AGX server. This call will block until
     * a frame is received, or until the AGX server closes the connection. In particular,
     * calling 'cancel' will not affect 'readSystem'.
     *
     * Each call to 'readSystem' should be followed by a call to 'writeSolution'.
     */
    agxData::Frame* readSystem( /*int timeout*/ ); // We might want a time out here.

    /**
     * Only for use in the read/write mode.
     * Send a  solve result back to the AGX server. Each call to 'writeSolution' should be
     * preceded with a call to 'readSystem'. How else should you know what to solve?
     */
    void writeSolution( agxData::Frame* solution );


    bool startSimulation();
    static bool startSimulation( const agxNet::IPAddress& server );

    bool stopSimulation();
    static bool stopSimulation( const agxNet::IPAddress& server );

    bool disableRemoteSolving();
    static bool disableRemoteSolving( const agxNet::IPAddress& server );

    agxNet::IPAddress enableRemoteSolving();
    static agxNet::IPAddress enableRemoteSolving( const agxNet::IPAddress& server );

  protected:
    ~RemoteSolverClient() {}


  private:
    agxNet::IPAddress m_serverAddress;
    agxNet::IPAddress m_remoteSolverAddress;

    agxNet::TcpSocketFrameReaderRef m_reader;
    agxNet::TcpSocketFrameWriterRef m_writer;
    FrameReceivedCallback m_callback;
    bool m_canceled;
  };



  AGX_FORCE_INLINE void RemoteSolverClient::cancel() { m_canceled = true; }

}
DOXYGEN_END_INTERNAL_BLOCK()

#endif
