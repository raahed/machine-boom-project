/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#pragma once

#include <agx/Logger.h>
#include <agx/version.h>

namespace agxUtil
{
  /**
   Register a callback to be called at the end of the current scope.

   The callback may not throw exceptions, std::terminate may be called
   immediately if it does.

   The callback should not take any parameters. The return value is ignored.

   A factory function, onScopeExit, is provided to make OnScopeExit using
   lambdas possible.

   Example usage:

     void myFunction()
     {
       FILE* myFile = fopen(....);
       const auto& closeFile = onScopeExit([myFile]() {
         fclose(myFile);
       });
       std::ignore = closeFile;
       // std::ignore is used to prevent warning about unused variable.


       // The rest of the function goes here, possibly with early returns or
       // exception throwing.
    }
   */
  template <typename Callback>
  class OnScopeExit
  {
  public:
    OnScopeExit(Callback&& callback)
      : m_callback(std::move(callback))
    {
    }

    OnScopeExit(OnScopeExit&& other)
      : m_callback(std::move(other.m_callback))
    {
    }

    OnScopeExit(const OnScopeExit&) = delete;
    OnScopeExit& operator=(const OnScopeExit&) = delete;

    ~OnScopeExit()
    {
      try
      {
        m_callback();
      }
      catch (...)
      {
        // What would be a sane thing to do here? Exceptions should not escape destructors.
        LOGGER_ERROR() << "Exception thrown from a OnScopeExit callback. This is not allowed." << LOGGER_ENDL();
      }
    }

  private:
    Callback m_callback;
  };

  template <typename Callback>
  OnScopeExit<Callback> onScopeExit(Callback&& callback)
  {
    return OnScopeExit<Callback>(std::forward<Callback>(callback));
  }
}
