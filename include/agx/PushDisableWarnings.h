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


/*
No include guard on purpose - this is for easier enabling/disabling of warnings
around includes of external header files.

ALWAYS REMEMBER to include in the following order:

...
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <externalHeaderFoo.h>
#include <externalHeaderBar.h>
...
#include <agx/PopDisableWarnings.h> // End of disabled warnings.
...
*/

// define AGX_WARNINGS_IN_EXTERNAL_HEADERS if warnings in external headers should be enabled
#ifndef AGX_WARNINGS_IN_EXTERNAL_HEADERS

#ifdef _MSC_VER
#  pragma warning(push, 0)
#  include <CodeAnalysis/Warnings.h>
#  pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#endif

#ifdef __clang__
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wcast-align"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wempty-body"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wunused-function"
#  pragma clang diagnostic ignored "-Woverloaded-virtual"
#  pragma clang diagnostic ignored "-Wnull-dereference"
#  pragma clang diagnostic ignored "-Wreturn-stack-address" // <osg/io_utils>
#  if __clang_major__ > 4
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#  else
#    pragma clang diagnostic ignored "-Wdeprecated"
#  endif
#  pragma clang diagnostic ignored "-Wall"
#  pragma clang diagnostic ignored "-Wextra"
#  pragma clang diagnostic ignored "-Wshadow"
#  pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__) && ( __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6 ) )
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wformat-nonliteral"
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wliteral-suffix"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC__ >= 9)
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#  pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif
#  pragma GCC diagnostic ignored "-Wliteral-suffix"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wunused-function"
#  pragma GCC diagnostic ignored "-Wunused-value"
#  pragma GCC diagnostic ignored "-Woverloaded-virtual"
#  pragma GCC diagnostic ignored "-Wall"
#  pragma GCC diagnostic ignored "-Wextra"
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#  pragma GCC diagnostic ignored "-Wundef"
#endif

#if defined(__APPLE__)
// OSG with deprecated OpenGL API
#define GL_SILENCE_DEPRECATION 1
#endif

#endif
