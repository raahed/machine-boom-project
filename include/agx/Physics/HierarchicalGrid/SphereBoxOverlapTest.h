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

#ifndef AGX_HIERARCHICALGRIDSPHEREBOXOVERLAPTEST_H
#define AGX_HIERARCHICALGRIDSPHEREBOXOVERLAPTEST_H

#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>

namespace agx
{
  
  #define AXIS_TEST(axis)                                                                                   \
  {                                                                                                         \
    Real diff = std::abs( localSpherePosition[axis] ) - boxHalfExtents[axis];                               \
    if (diff > 0) {                                                                                         \
      if (diff > sphereRadius)                                                                              \
        return false; /* Early out */                                                                       \
      isOutside = true;                                                                                     \
      pointOnBox[axis] = (localSpherePosition[axis] > 0) ? boxHalfExtents[axis] : -boxHalfExtents[axis];    \
    }                                                                                                       \
  }
  

  inline bool sphereBoxOverlapTest
  (
    Real sphereRadius, const Vec3& spherePosition, 
    const Vec3& boxHalfExtents, const AffineMatrix4x4 &boxInvTransform
  )
  {

    // This is easy. get the sphere center `p' relative to the box, and then clip
    // that to the boundary of the box (call that point `q'). if q is on the
    // boundary of the box and |p-q| is <= sphere radius, they touch.
    // if q is inside the box, the sphere is inside the box, so set a contact
    // normal to push the sphere to the closest box face.

    // all collision detection takes place in box's frame
    Vec3 localSpherePosition = spherePosition * boxInvTransform;
    Vec3 pointOnBox = localSpherePosition;

    bool isOutside = false;
    // Unrolled loop
    AXIS_TEST(0)
    AXIS_TEST(1)
    AXIS_TEST(2)
    
    if (isOutside) {
      Vec3 normal = localSpherePosition - pointOnBox;
      Real distance2 = normal.length2();
      if (distance2 > sphereRadius * sphereRadius)
        return false; //sphere completely outside of box
    }
    
    return true;
  }
  
}


#endif /* AGX_HIERARCHICALGRIDSPHEREBOXOVERLAPTEST_H */
