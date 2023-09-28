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

#pragma once

#include <agx/config.h>

#ifdef _WIN32
# if AGX_DYNAMIC() &&  defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__) || defined( __BCPLUSPLUS__)  || defined( __MWERKS__)
#  if defined( AGX_LIBRARY_STATIC )
#    define AGXVEHICLE_EXPORT
#  elif defined( AGXVEHICLE_LIBRARY )
#    define AGXVEHICLE_EXPORT   __declspec(dllexport)
#  else
#    define AGXVEHICLE_EXPORT   __declspec(dllimport)
#  endif
# else
#  define AGXVEHICLE_EXPORT
# endif
#else
  // Non Win32
  #if __GNUC__ >= 4
    #define AGXVEHICLE_EXPORT __attribute__ ((visibility("default")))
  #else
    #define AGXVEHICLE_EXPORT
  #endif
#endif
