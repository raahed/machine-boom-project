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

#ifndef AGX_BUILD_FLAGS_H
#define AGX_BUILD_FLAGS_H

#include <agx/agxCore_export.h>

namespace agx
{
  AGXCORE_EXPORT extern const char* const BUILD_TYPE;
  AGXCORE_EXPORT extern const char* const CPP_BUILD_FLAGS;
  AGXCORE_EXPORT extern const char* const C_BUILD_FLAGS;
}

// Backward compability macro definitions
#ifndef AGX_BUILD_TYPE
#define AGX_BUILD_TYPE agx::BUILD_TYPE
#endif

#ifndef AGX_CPP_BUILD_FLAGS
#define AGX_CPP_BUILD_FLAGS agx::CPP_BUILD_FLAGS
#endif

#ifndef AGX_C_BUILD_FLAGS
#define AGX_C_BUILD_FLAGS agx::C_BUILD_FLAGS
#endif

#endif
