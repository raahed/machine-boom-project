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


namespace agxSDK
{
  class Simulation;
}

namespace agxNet
{

  /// Class for receiving serialized agxSDK::Simulation:s from remote host
  class AGXPHYSICS_EXPORT CoSimulationClient : public agxNet::CoSimulationBase, public agx::Referenced
  {
  public:
    /**
    Constructor.
    \param address - Address of server running a simulation
    */
    CoSimulationClient( const IPAddress& address );

    agx::String getMessage();

    bool checkConnection();

    void setTimeStep(double timeStep);

    /**
    * Preloads the scene, parsing the file and obtaining information about input and output sizes.
    * \param filename The name of the script (python or lua)  file.
    *        Has to have a buildScene(sim, app) function.
    * \retval Has the loading succeeded?
    */
    bool preloadScript(const agx::String& filename);

    bool initInput(const agx::RealVector& input, agx::RealVector& output);

    bool stepInput(const agx::RealVector& input, agx::RealVector& output);

    /// Is the client still waiting for connection?
    bool isWaitingForConnection() const;

    virtual bool waitForConnection();

  protected:

    bool connectToServer();

    virtual void handleConnectionDrop();

    virtual ~CoSimulationClient();

  private:
    IPAddress m_address;
    bool m_isWaitingForConnection;
    agx::Timer stepTimer;

  };

}
