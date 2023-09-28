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

#ifndef AGX_PLUGINMACROS_H
#define AGX_PLUGINMACROS_H

#include <agx/agxCore_export.h>
#include <agx/String.h>

namespace agx
{
  typedef void* PluginHandle;
  typedef void(*FunctionHandle)();

  /// Opens a plugin. Corresponds to LoadLibrary or dlopen.
  AGXCORE_EXPORT PluginHandle OpenPlugin(const char* path);

  /**
  Closes a plugin. Corresponds to FreeLibrary or dlclose.
  \retval true on success, and false on error.
  */
  AGXCORE_EXPORT bool ClosePlugin(PluginHandle handle);


  AGXCORE_EXPORT FunctionHandle MapPluginSymbol(PluginHandle handle, const char *symbolName);


  /// Get plugin error.
  AGXCORE_EXPORT agx::String GetPluginError();

}

#endif /* _AGX_PLUGINMACROS_H_ */
