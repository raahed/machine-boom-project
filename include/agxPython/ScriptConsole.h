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
#ifndef AGXPYTHON_SCRIPTCONSOLE_H
#define AGXPYTHON_SCRIPTCONSOLE_H 1

#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_PYTHON()

#include <agxPython/export.h>

#include <agx/Referenced.h>
#include <agx/String.h>

#include <string>

namespace agxPython
{

  class AGXPYTHON_EXPORT ScriptConsole : public agx::Referenced
  {
  public:

    /**
    Console buffer redirection require the use of an embedded Python
    module to be initialized with the global interpreter state. This
    must be called before Py_InitializeEx() which is in turn called by
    agxPython::ScriptManager::init().
    */
    static bool EmbedIntoPython();

    ScriptConsole();

    /**
    Override to redirect output to stdout from default cout. Line-endings
    are system specific when they appear in the buffer string.
    Strings are UTF-8 encoded.
    */
    virtual void OnStdout(const agx::String& buffer);

    /**
    Override to redirect output to stdout from default cerr. Line-endings
    are system specific when they appear in the buffer string.
    Strings are UTF-8 encoded.
    */
    virtual void OnStderr(const agx::String& buffer);

    void write(const std::string& s, void *type);

    void init();

    void set_stdout();
    void reset_stdout();

    void set_stderr();
    void reset_stderr();

  protected:

    virtual ~ScriptConsole();

  private:

    void *m_stdout;
    void *m_stdout_saved;

    void *m_stderr;
    void *m_stderr_saved;

  };


}



#endif

#endif
