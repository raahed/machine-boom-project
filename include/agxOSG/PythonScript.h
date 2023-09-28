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

#ifndef AGXOSG_PYTHONSCRIPT_H
#define AGXOSG_PYTHONSCRIPT_H

#include <agx/config/AGX_USE_PYTHON.h>
#include <agxOSG/export.h>
#include <string>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgDB/ReadFile>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Math.h>
#include <agxSDK/Simulation.h>

namespace osg {
  class Node;
}

namespace agxSDK
{
  class Simulation;
  class Assembly;
}

namespace agxOSG
{
  class ExampleApplication;
  class Group;

#if AGX_USE_PYTHON()

  DOXYGEN_START_INTERNAL_BLOCK()
  bool executePythonScriptImpl(const std::string& filenameOrScript,
    const std::string& function,
    agxSDK::Simulation *simulation,
    agxOSG::ExampleApplication* application,
    osg::Group *root,
    bool filenameGiven,
    std::string& errorMessage);
  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  Execute a named function in a python script file. In contrast to how things are
  done with Lua scripts, this function should not receive any arguments but instead
  retrieve the Simulation, ExampleApplication and OSG group root node objects from
  the ScriptContext instance retrievable from the agxPython module, e.g in Python:

  --

    sim, app, group = (agxPython.scriptContext().getSimulation(),
                       agxPython.scriptContext().getApplication(),
                       agxPython.scriptContext().getGroup())

  --

  If agxPython.scriptContext().getSimulation() returns None, the script is not run
  from AGX. Otherwise, the current Simulation object is returned. AGX Python scripts
  should end with an if-statement to check if it is None and separately deal with that
  scenario.

  The function should not return anything (it will not be handled).

  The script file will be "reloaded" every time, so changes in the file between two calls will be reflected.
  \param filename - path to the python script
  \param function - name of function to be called (from global scope)
  \param simulation - pointer to a valid simulation.
  \param application - pointer to the current ExampleApplication being used (could very well be nullptr).
  \param root - pointer to a scene graph root where to put the graphics nodes (can also be nullptr).
  */
  AGXOSG_EXPORT bool executePythonScript(const std::string& filename,
    const std::string& function,
    agxSDK::Simulation *simulation,
    agxOSG::ExampleApplication* application,
    osg::Group *root);

  /**
  Execute a named function in a python script. This function should take no arguments and exist in
  global scope:

  function()

  The function should not return anything (it will not be handled).

  \param script - the python script string
  \param function - name of function to be called (from global scope)
  \param simulation - pointer to a valid simulation.
  \param application - pointer to the current ExampleApplication being used (could very well be nullptr).
  \param root - pointer to a scene graph root where to put the graphics nodes (can also be nullptr).
  \param errorMessage - Errors will be appended to this string.

  */
  AGXOSG_EXPORT bool executePythonScriptString(const std::string& script,
    const std::string& function,
    agxSDK::Simulation *simulation,
    agxOSG::ExampleApplication* application,
    osg::Group *root,
    std::string& errorMessage);

  /**
  Checks if there were any error messages when a lua script was executed.

  \return The last error message, or the empty string if no error.
  */
  AGXOSG_EXPORT std::string readPythonScriptError();

  /**
  Resets the script error to the empty string.
  */
  AGXOSG_EXPORT void resetPythonScriptError();

#endif
}

#endif /* AGXOSG_PYTHONSCRIPT_H */
