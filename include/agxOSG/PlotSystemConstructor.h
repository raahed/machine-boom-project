/*
Copyright 2007-2023. Algoryx Simulation AB.

All AgX source code, intellectual property, documentation, sample code,
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

#ifndef AGXOSG_PLOTSYSTEM_CONSTRUCTOR
#define AGXOSG_PLOTSYSTEM_CONSTRUCTOR

#include <agxOSG/export.h>
#include <agx/config/AGX_USE_AGXCALLABLE.h>
#include <agx/config/AGX_USE_WEBPLOT.h>
#include <agxIO/ReaderWriter.h>
#include <agxIO/FileSystem.h>
#include <agx/String.h>

namespace agxOSG
{
  // Class used to initialize a plotsystem from supplied xml
  class AGXOSG_EXPORT PlotsystemConstructor
  {
  public:
    static bool constructPlotsystemFromXML(const agx::String& bindingsFile, agxSDK::Simulation* simulation
#if AGX_USE_WEBPLOT()
                                           , bool useWebplot
#endif
      );
  };
}


#endif
