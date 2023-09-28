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

#ifndef AGX_INITIALIZER_H
#define AGX_INITIALIZER_H

#include <agx/agx.h>
#include <agx/Integer.h>
#include <agx/Callback.h>

namespace agx
{
  /**
  AGX_STATIC_INITIALIZER is used to run code statically as agx is loaded. Normally before main is entered. Example usage:

  AGX_STATIC_INITIALIZER()
  {
    printf("Init!");
  }
  */
  #define AGX_STATIC_INITIALIZER()                                                                                          \
  static void AGX_CONCAT(__agx_static_init, __LINE__)(void);                                                                \
  static agx::StaticInitializer AGX_CONCAT(__agx_static_init_dispatch, __LINE__)(&AGX_CONCAT(__agx_static_init, __LINE__)); \
  static void AGX_CONCAT(__agx_static_init, __LINE__)(void)

  // Do not use class directly
  class AGXCORE_EXPORT StaticInitializer
  {
  public:
    typedef void (*Implementation)(void);

    StaticInitializer(Implementation implementation);
  };


  //-----------------------------------------------------------------------------------------------------
  #define AGX_STATIC_SHUTDOWN()                                                                                                  \
  static void AGX_CONCAT(__agx_static_shutdown, __LINE__)(void);                                                                 \
  static agx::StaticShutdown AGX_CONCAT(__agx_static_shutdown_dispatch, __LINE__)(&AGX_CONCAT(__agx_static_shutdown, __LINE__)); \
  static void AGX_CONCAT(__agx_static_shutdown, __LINE__)(void)

  // Do not use class directly
  class AGXCORE_EXPORT StaticShutdown
  {
  public:
    typedef void (*Implementation)(void);

    StaticShutdown(Implementation implementation);
    virtual ~StaticShutdown();

  private:
    Implementation m_implementation;
  };


  /**
  Register a function to be called on all subsequent executions of agx::init().
  If AGX is currently initialized, then the provided function is called
  immediately.
  */
  #define AGX_INIT_CALLBACK() \
    static void AGX_CONCAT(__agx_init_callback, __LINE__)(); \
    static agx::InitCallback AGX_CONCAT(__agx_init_callback_dispatch, __LINE__)(&AGX_CONCAT(__agx_init_callback, __LINE__)); \
    static void AGX_CONCAT(__agx_init_callback, __LINE__)()

  /// Do not use class directly. Use AGX_INIT_CALLBACK instead.
  class AGXCORE_EXPORT InitCallback
  {
  public:
    using Implementation = void(*)();
    explicit InitCallback(Implementation implementation);

  private:
    friend void agxCore::init();
    static void triggerInit();

  private:
    Implementation m_implementation;
    InitCallback* m_next;
  };


  //-----------------------------------------------------------------------------------------------------

  /**
  AGX_SHUTDOWN_CALLBACK is used to run code when agx::shutdown is called. Note the difference to AGX_SHUTDOWN_CALLBACK,
  which is only called once when the agx library is unloaded, agx::shutdown on the other hand can be called many times.
  Example usage:

  AGX_SHUTDOWN_CALLBACK()
  {
    printf("agx::shutdown called!\n");
  }
  */
  #define AGX_SHUTDOWN_CALLBACK()                                                                                                    \
  static void AGX_CONCAT(__agx_shutdown_callback, __LINE__)(void);                                                                   \
  static agx::ShutdownCallback AGX_CONCAT(__agx_shutdown_callback_dispatch, __LINE__)(&AGX_CONCAT(__agx_shutdown_callback, __LINE__)); \
  static void AGX_CONCAT(__agx_shutdown_callback, __LINE__)(void)


  // Do not use class directly
  class AGXCORE_EXPORT ShutdownCallback
  {
  public:
    typedef void (*Implementation)(void);

    ShutdownCallback(Implementation implementation);

  private:
    friend void agxCore::shutdown();
    static void triggerShutdown();

  private:
    Implementation m_implementation;
    ShutdownCallback *m_next;
  };


}



#endif /* _AGX_INITIALIZER_H_ */
