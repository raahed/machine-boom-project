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

#ifndef AGXOSG_LUASCRIPT_H
#define AGXOSG_LUASCRIPT_H

#include <agx/config/AGX_USE_LUA.h>
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

#if AGX_USE_LUA()

  DOXYGEN_START_INTERNAL_BLOCK()
  bool executeScriptImpl(const std::string& filenameOrScript,
                         const std::string& function,
                         agxSDK::Simulation *simulation,
                         agxOSG::ExampleApplication* application,
                         osg::Group *root,
                         bool filenameGiven,
                         std::string& errorMessage);
  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  Execute a named function in a lua script file. This function should take 3 arguments:
  <function>(agxSDK::Simulation *simulation, agxOSG::ExampleApplication *application, osg::Group *root))

  The function should not return anything (it will not be handled).

  It is up to the lua function to check if any of the arguments are != 0.
  The script file will be "reloaded" every time, so changes in the file between two calls will be reflected.
  \param filename - path to the lua script
  \param function - name of function to be called (in global scope)
  \param simulation - pointer to a valid simulation.
  \param application - pointer to the current ExampleApplication being used (could very well be null).
  \param root - pointer to a scene graph root where to put the graphics nodes.

  */
  AGXOSG_EXPORT bool executeScript(const std::string& filename,
                                   const std::string& function,
                                   agxSDK::Simulation *simulation,
                                   agxOSG::ExampleApplication* application,
                                   osg::Group *root);

  /**
  Execute a named function in a lua script. This function should take 3 arguments:
  <function>(agxSDK::Simulation *simulation, agxOSG::ExampleApplication *application, osg::Group *root))

  The function should not return anything (it will not be handled).

  It is up to the lua function to check if any of the arguments are != 0.
  \param script - the lua script string
  \param function - name of function to be called (in global scope)
  \param simulation - pointer to a valid simulation.
  \param application - pointer to the current ExampleApplication being used (could very well be null).
  \param root - pointer to a scene graph root where to put the graphics nodes.

  */
  AGXOSG_EXPORT bool executeScriptString(const std::string& script,
                                         const std::string& function,
                                         agxSDK::Simulation *simulation,
                                         agxOSG::ExampleApplication* application,
                                         osg::Group *root,
                                         std::string& errorMessage);

  /**
  Checks if there were any error messages when a lua script was executed.

  \return The last error message, or the empty string if no error.
  */
  AGXOSG_EXPORT std::string readScriptError();

  /**
  Resets the script error to the empty string.
  */
  AGXOSG_EXPORT void resetScriptError();

#endif
}

#endif /* AGXOSG_LUASCRIPT_H */
