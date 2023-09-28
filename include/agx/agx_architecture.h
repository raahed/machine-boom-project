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

#ifndef AGX_ARCHITECTURE_H
#define AGX_ARCHITECTURE_H

// Visual studio
#if defined(_MSC_VER)
#  if defined(_M_IX86) // 32bit compiler
#    define AGX_64BIT_ARCHITECTURE 0
#  elif defined(_M_X64) // 64bit compiler
#    define AGX_64BIT_ARCHITECTURE 1
#  else
#    error "Architecture not defined"
#  endif
#elif defined(SWIG)
#  if defined(_M_X64) || defined(__LP64__) || defined(AGX_SWIG_64BIT)
#    define AGX_64BIT_ARCHITECTURE 1
#  else
#    define AGX_64BIT_ARCHITECTURE 0
#  endif
#else
#  if defined(__LP64__) // 64bit compiler
#    define AGX_64BIT_ARCHITECTURE 1
#  else // 32bit compiler
#    define AGX_64BIT_ARCHITECTURE 0
#  endif
#endif
#endif // AGX_ARCHITECTURE_H

