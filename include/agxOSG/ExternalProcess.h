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

#ifndef AGXOSG_EXTERNALPROCESS_H
#define AGXOSG_EXTERNALPROCESS_H

#include <agxOSG/export.h>
#include <agx/Referenced.h>
#include <thread>
#include <memory>
#include <atomic>
#include <agx/String.h>

/// \cond INTERNAL_DOCUMENTATION
namespace detail
{
  class Process;
}
/// \endcond

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(ExternalProcess);
  /**
  Cross-platform utility class for wrapping and launching external processes.
  */
  class AGXOSG_EXPORT ExternalProcess : public agx::Referenced
  {
  public:
    /// Default constructor
    ExternalProcess(const agx::String& processTag = "");

    /*
    Launch an external process with a shell command.
    \param command The shell command to be executed
    \param workingDir The working directory where the process will be executed
    \return True if process is still running
    */
    bool startProcess(const agx::StringVector& commandVector, const agx::String& workingDir);

    /**
    Will wait until the external process has exited. Return the exit code when the process is done.
    \return The exit code of the process
    */
    int waitUntilExit();

    /**
    Will try to close the external process and wait for it's exit
    \return The exit code of the process.
    */
    int closeAndWaitForExit();

    /// Will kill the external process by forcing a close.
    void killProcess();

    /// Return true if the process is still running. False it not.
    bool isRunning() const;

    ExternalProcess(const ExternalProcess&) = delete;
  protected:
    virtual ~ExternalProcess();

    bool startProcess(const agx::String& command, const agx::String& workingDir);

  private:
    const agx::String                m_processTag;
    std::unique_ptr<detail::Process> m_externalProcess;
    std::thread                      m_processThread;
    std::atomic_bool                 m_processIsRunning;
    int                              m_exitStatus;
  };
}

#endif
