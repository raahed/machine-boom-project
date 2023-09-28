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
#ifndef AGXPYTHON_SCRIPTIDEINTERPRETER_H
#define AGXPYTHON_SCRIPTIDEINTERPRETER_H 1

#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_PYTHON()

#include <agxPython/export.h>
#include <agx/Referenced.h>



namespace agxPython
{
  class ScriptIDE;

  class AGXPYTHON_EXPORT ScriptIDEInterpreter : public agx::Referenced
  {

  public:

    friend class ScriptIDE;

    ScriptIDEInterpreter();

    virtual void onException() = 0;

    virtual void onStarted() = 0;

    virtual void onSuspended() = 0;

    virtual void onAbort() = 0;

    virtual void onBreakpoint(int line) = 0;

    virtual bool isEnabled();

    virtual void setEnabled(bool enable = true);

    void start(bool restart);

    void stop();

  protected:

    virtual ~ScriptIDEInterpreter();

  private:

    ScriptIDE *m_ide;

    bool m_enabled;

  };


}


#endif

#endif
