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

#ifndef AGXHYDRAULICS_EXPORT_H
#define AGXHYDRAULICS_EXPORT_H

#include <agx/config.h>

#ifdef _WIN32
# if AGX_DYNAMIC() &&  defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGX_LIBRARY_STATIC )
#    define AGXHYDRAULICS_EXPORT
#  elif defined( AGXHYDRAULICS_LIBRARY )
#    define AGXHYDRAULICS_EXPORT   __declspec(dllexport)
#  else
#    define AGXHYDRAULICS_EXPORT   __declspec(dllimport)
#  endif
# else
#  define AGXHYDRAULICS_EXPORT
# endif
#else
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXHYDRAULICS_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXHYDRAULICS_EXPORT
  #endif
#endif

/**
The agxHydraulics namespace contains classes for modeling hydraulic circuits
within the power line framework.
*/
namespace agxHydraulics
{
  /**
  The detail namespace contains helper classes that user code should not create
  directly but may receive as return values from accessor methods or factory
  functions.
  */
  namespace detail
  {
  }
}

#endif
