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
#  pragma warning(pop)
#endif

#ifdef __clang__
#  pragma clang diagnostic pop
#elif defined(__GNUC__) && ( __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6 ) )
#  pragma GCC diagnostic pop
#endif

#if defined(__APPLE__)
// OSG with deprecated OpenGL API
#undef GL_SILENCE_DEPRECATION
#endif

#endif
