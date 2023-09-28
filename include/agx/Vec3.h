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


#ifndef AGX_VEC3_H
#define AGX_VEC3_H

#include <agx/Vec3Template.h>
#include <agxData/Type.h>

namespace agx
{

  /**
  The object holding 3 dimensional vectors and providing basic arithmetic.

  The 3D vector class holds an internal representations of vectors which
  can include padding and so on.  At this level of the API, the object is
  provided for user convenience.  In the internal representation used by
  the solvers, 3D vectors may be concatenated into larger vector blocks
  as seems fit.
  */
  typedef Vec3T<Real> Vec3;


  typedef Vec3T<Real32> Vec3f;
  typedef Vec3T<Real64> Vec3d;


  typedef Vec3T<Int> Vec3i;
  typedef Vec3T<UInt> Vec3u;

  typedef Vec3T<Int8> Vec3i8;
  typedef Vec3T<UInt8> Vec3u8;
  typedef Vec3T<Int16> Vec3i16;
  typedef Vec3T<UInt16> Vec3u16;
  typedef Vec3T<Int32> Vec3i32;
  typedef Vec3T<UInt32> Vec3u32;
  typedef Vec3T<Int64> Vec3i64;
  typedef Vec3T<UInt64> Vec3u64;

#ifndef SWIG
  const Vec3 X_AXIS( 1.0, 0.0, 0.0 );
  const Vec3 Y_AXIS( 0.0, 1.0, 0.0 );
  const Vec3 Z_AXIS( 0.0, 0.0, 1.0 );
#endif

  // Hash function
  AGX_FORCE_INLINE agx::UInt32 hash(agx::UInt32 x, agx::UInt32 y, agx::UInt32 z)
  {
    const agx::UInt32 p1 = 73856093UL;
    const agx::UInt32 p2 = 19349663UL;
    const agx::UInt32 p3 = 83492791UL;

    return (p1 * x) ^ (p2 * y) ^ (p3 * z);
  }

  template <typename T>
  struct HashFn< Vec3T<T> >
  {
    AGX_FORCE_INLINE agx::UInt32 operator()(const Vec3T<T>& id) const
    {
      return agx::hash((agx::UInt32)id[0], (agx::UInt32)id[1], (agx::UInt32)id[2]);
    }
  };

  AGX_FORCE_INLINE Vec3i calculateCellId(Real x, Real y, Real z, Real invCellSize)
  {
    agx::Vec3i cell;
    cell[0] = (Vec3i::Type)(x * invCellSize);
    cell[1] = (Vec3i::Type)(y * invCellSize);
    cell[2] = (Vec3i::Type)(z * invCellSize);

    if (x < Real(0.0)) cell[0]--;
    if (y < Real(0.0)) cell[1]--;
    if (z < Real(0.0)) cell[2]--;

    return cell;
  }

  AGX_FORCE_INLINE Vec3i calculateCellId(Vec3 position, Real invCellSize)
  {
    return calculateCellId(position[0], position[1], position[2], invCellSize);
  }

}   // end of namespace agx


AGX_TYPE_BINDING(agx::Vec3f, "Vec3")
AGX_TYPE_BINDING(agx::Vec3d, "Vec3")

AGX_TYPE_BINDING(agx::Vec3i8 , "Vec3i")
AGX_TYPE_BINDING(agx::Vec3i16, "Vec3i")
AGX_TYPE_BINDING(agx::Vec3i32, "Vec3i")
AGX_TYPE_BINDING(agx::Vec3i64, "Vec3i")

AGX_TYPE_BINDING(agx::Vec3u8 , "Vec3u")
AGX_TYPE_BINDING(agx::Vec3u16, "Vec3u")
AGX_TYPE_BINDING(agx::Vec3u32, "Vec3u")
AGX_TYPE_BINDING(agx::Vec3u64, "Vec3u")

#endif
