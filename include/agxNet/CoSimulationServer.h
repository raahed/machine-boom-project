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

#include <agxNet/CoSimulationBase.h>
#include <agx/agx_vector_types.h>

#include <agx/Thread.h>

namespace agxSDK
{
  class Simulation;
}

namespace agxNet
{

  /// Class for sending serialized agxSDK::Simulation over to a client
  class AGXPHYSICS_EXPORT CoSimulationServer : public agxNet::CoSimulationBase, public agx::BasicThread, public agx::Referenced
  {
  public:

    void run();

    /// Shutdown the communication threads, will cause this client to be non-usable any more.
    void shutdown();


    enum ScriptMode
    {
      NO_SCRIPT=0,
      PYTHON_SCRIPT=1,
    };

    void setScriptMode(ScriptMode);
    ScriptMode getScriptMode() const;

    /**
    Constructor.
    \param port - The port where the server will listen for connections.
    */
    CoSimulationServer( uint16_t port );

    /**
    Enable/disable compression of transmitted data
    \param flag - if true, data will be compressed
    */
    void setEnableCompression( bool flag ) { m_useCompression = flag; }

    /**
    \return true if compression is enabled
    */
    bool getEnableCompression( ) const { return m_useCompression; }

    const agx::String& getFilename();

    Header::Mode receiveInstructions( agxSDK::Simulation *simulation );
    void setInstructionStatus( Header::Mode mode, const agx::String& message );

    bool applyInitInputs(agxSDK::Simulation *simulation );
    bool obtainInitOutputs(agxSDK::Simulation* simulation );

    bool isShutdown();

  protected:

    virtual void waitForDataAndTransfer();
    virtual bool waitForConnection();
    void dropConnection(bool shouldShutdown) override;

    virtual ~CoSimulationServer();

  private:
    void handlePreloadScript();
    void handleSetTimeStep();
    void handleStepInput();
    void handleInitInput();
    void handleNop();

    void sendReplyPacket();

    TCPServerSocketRef m_serverSocket;
    uint16_t m_port;
    bool m_useCompression;

    agx::Block m_hasInstructions;

    Header::Mode m_currentInstruction;
    Header::Mode m_replyInstruction;
    agx::String m_replyMessage;
    agx::String m_filename;
    double m_timeStep;

    agx::RealVector m_inputs;
    agx::RealVector m_outputs;

    agx::RealVector m_initInputs;
    agx::RealVector m_initOutputs;
    agx::Timer stepTimer;
    ScriptMode m_scriptMode;

  };

}
