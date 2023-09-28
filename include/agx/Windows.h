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

#ifndef AGX_WINDOWS_H
#define AGX_WINDOWS_H

// Include file to get the include of windows.h right, which is a bit tricky.
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#  include <windows.h>
#  include <agx/PopDisableWarnings.h> // End of disabled warnings.
#endif

#ifdef far // Defined in WinDef.h
#  undef far
#endif

#ifdef near // Defined in WinDef.h
#  undef near
#endif

#ifdef FAR  // Defined in WinDef.h; has to be defined again
#  undef FAR
#  define FAR
#endif

#ifdef NEAR // Defined in WinDef.h; has to be defined again
#  undef NEAR
#  define NEAR
#endif


#endif

