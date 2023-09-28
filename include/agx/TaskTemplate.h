/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_TASKTEMPLATE_H
#define AGX_TASKTEMPLATE_H

#include <agx/agx.h>
#include <agx/Task.h>
// #include <agx/TaskManager.h>
#include <agx/TaskExport.h>
//#include <agx/Component.h>
#include <agx/TaskManager.h>
#include <agx/Model.h>

typedef agx::Task * (*TaskConstructor)(agx::Device *device);

#define AGX_PRE_JOB
#define AGX_POST_JOB

#define AGX_TASK(_Name, _Path)                                                                \
namespace                                                                              \
{                                                                                      \
  agx::TaskPlugin plugin_ ## _Name(_Path, agx::Device::CPU);                           \
  agx::TaskPluginImplementationT<_Name> plugin_implementation(&plugin_ ## _Name);       \
}

#if 0
                                                         \
extern "C"                                               \
{                                                        \
                                                         \
AGXTASK_EXPORT const char *name()                        \
{                                                        \
  return #_Name;                                         \
}                                                        \
                                                         \
                                                         \
AGXTASK_EXPORT agx::Device::Type deviceType()            \
{                                                        \
  return _DeviceType;                                    \
}                                                        \
                                                         \
AGXTASK_EXPORT agx::Task *create(agx::Device *device, agx::UInt implementation) \
{                                                        \
  return implementation == 0 ? new ::_Name(device) : nullptr;  \
}                                                        \
                                                         \
AGXTASK_EXPORT void destroy(agx::Task * /*task*/)             \
{                                                        \
  /*delete task; */                                      \
}                                                        \
}

#define AGX_OVERLOADED_TASK(_Name, _DeviceType) \
                                                         \
extern "C"                                               \
{                                                        \
                                                         \
AGXTASK_EXPORT const char *name()                        \
{                                                        \
  return #_Name;                                         \
}                                                        \
                                                         \
                                                         \
AGXTASK_EXPORT agx::Device::Type deviceType()            \
{                                                        \
  return _DeviceType;                                    \
}                                                        \
                                                         \
AGXTASK_EXPORT agx::Task *create(agx::Device *device, agx::UInt implementation) \
{                                                        \
  return constructors[implementation] ? constructors[implementation](device) : nullptr; \
}                                                        \
                                                         \
AGXTASK_EXPORT void destroy(agx::Task * /*task*/)             \
{                                                        \
  /*delete task; */                                      \
}                                                        \
}
#endif


#endif /* _AGX_TASKTEMPLATE_H_ */

