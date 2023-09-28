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

#ifndef AGXIO_WIN32_TOOLS_H
#define AGXIO_WIN32_TOOLS_H

#include <agx/config.h>
#include <agx/String.h>


namespace agxIO
{
  /**
  Get a string value from the registry written by the Installer
  \param key - Name of the key
  \param value - At success, the value will contain the string content of the key.
  \return true if the key exist and is of type string and we have read access to it.
  */
  AGXCORE_EXPORT bool getRegisterValue( const char *key, agx::String& value );
}
#endif

