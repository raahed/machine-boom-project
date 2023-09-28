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

#ifndef AGXLUA_GARBAGELISTENER_H
#define AGXLUA_GARBAGELISTENER_H

#include <agxLua/export.h>
#include <agxSDK/StepEventListener.h>

namespace agxLua
{
  /**
  Simulation is a class that bridges the collision space agxCollide::Space and the dynamic simulation system
  agx::DynamicsSystem.
  */
  class AGXLUA_EXPORT GarbageListener : public agxSDK::StepEventListener
  {
  public:
    GarbageListener();
    void addNotification();
    void pre( const agx::TimeStamp& t);
  };
} // namespace agxLua
#endif
