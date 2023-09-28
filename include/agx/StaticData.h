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

#ifndef AGX_STATICDATA_H
#define AGX_STATICDATA_H

#include <agx/agx.h>
#include <agx/agxCore_export.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4505) // warning C4505: 'X' : unreferenced local function has been removed
#endif

/**
Static data helper to avoid static data initialization order problems (before main). Local function member
variable is allocated on demand. StaticDataInitializer is used to force initialization before main to avoid
potential thread data race conditions.
Usage:

AGX_STATIC_DATA(s_var, MyClass);
...
s_var().doSomething();
*/
#define AGX_STATIC_DATA(_Name, _Type)                                              \
static _Type& _Name()                                                              \
{                                                                                  \
  static _Type s_instance;                                                         \
  return s_instance;                                                               \
}                                                                                  \
static agx::StaticDataInitializer AGX_CONCAT(__agx_static_data_initializer, _Name)(_Name)


namespace agx
{
  // Only used implicitly via AGX_STATIC_DATA
  class StaticDataInitializer
  {
  public:
    template <typename FuncT>
    StaticDataInitializer(FuncT f)
    {
      f();
    }
  };

}

#ifdef _MSC_VER
  #pragma warning(pop)
#endif
#endif /* _AGX_STATICDATA_H_ */
