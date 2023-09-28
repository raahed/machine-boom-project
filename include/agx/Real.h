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

#ifndef AGXDATA_REAL_H
#define AGXDATA_REAL_H

#include <agx/config.h>
#include <agx/agxCore_export.h>
#include <limits>

namespace agx
{
  /**
   The data type for basic floating point representation.

   The floating point representation used here is used in passing data to
   and from the user.  It is not necessarily identical to the floating
   point representation used inside the solver or in other parts of the
   internals.
  */

  enum RealModeEnum
  {
    REAL_32BIT,
    REAL_64BIT
  };

  static const RealModeEnum RealMode = REAL_64BIT;
  typedef double Real;

  typedef float Real32;
  typedef double Real64;

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

  AGXCORE_EXPORT extern const Real RealMax;
  AGXCORE_EXPORT extern const Real RealMin;
  AGXCORE_EXPORT extern const Real RealEpsilon;

  AGXCORE_EXPORT extern const Real32 Real32Max;
  AGXCORE_EXPORT extern const Real32 Real32Min;
  AGXCORE_EXPORT extern const Real32 Real32Epsilon;

  AGXCORE_EXPORT extern const Real64 Real64Max;
  AGXCORE_EXPORT extern const Real64 Real64Min;
  AGXCORE_EXPORT extern const Real64 Real64Epsilon;
}


#endif /* _AGXDATA_REAL_H_ */
