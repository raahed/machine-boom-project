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

#ifndef AGX_EULERCONVENTION_H
#define AGX_EULERCONVENTION_H

#include <agx/agxCore_export.h>

namespace agx
{
  class AGXCORE_EXPORT EulerConvention
  {
  public:
    /**
    The 24 Euler conventions.   The names of the enums refer to the sequence
    of rotations being performed, e.g., about X, then Y, then Z.  The last
    letter refers to whether or not we are referring to the static (or fixed)
    axes (s), or to the rotated axes (r).
    */
    enum Convention: agx::Int32 {
      /// the conventions with static axes
      XYZs = 0, XYXs = 2, XZYs = 4, XZXs = 6, YZXs = 8, YZYs = 10, YXZs = 12,
      YXYs = 14, ZXYs = 16, ZXZs = 18, ZYXs = 20, ZYZs = 22,
      /// the conventions with rotated axes
      ZYXr = 1, XYXr = 3, YZXr = 5, XZXr = 7, XZYr = 9, YZYr = 11,
      ZXYr = 13, YXYr = 15, YXZr = 17, ZXZr = 19, XYZr = 21, ZYZr = 23,
      LAST,
      FIRST = 0,                  // marker for last position
      BAD = -1,                    // place holder for bad values
      DEFAULT_CONVENTION = XYZs //!< The default Euler convention used by agx.

    };

    // Need this to make tolua work. It requires a constructor
    EulerConvention() {}

  };
} // namespace agx

#endif
