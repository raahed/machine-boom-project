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

#ifndef AGX_VEC2_H
#define AGX_VEC2_H

#include <agx/Vec2Template.h>
#include <agxData/Type.h>

namespace agx
{

  /**
  The object holding 2 dimensional vectors and providing basic arithmetic.

  The 2D vector class holds an internal representations of vectors which
  can include padding and so on.  At this level of the API, the object is
  provided for user convenience.  In the internal representation used by
  the solvers, 2D vectors may be concatenated into larger vector blocks
  as seems fit.
  */
  typedef Vec2T<Real> Vec2;
  typedef Vec2T<Real32> Vec2f;
  typedef Vec2T<Real64> Vec2d;


  typedef Vec2T<Int> Vec2i;
  typedef Vec2T<Int8> Vec2i8;
  typedef Vec2T<Int16> Vec2i16;
  typedef Vec2T<Int32> Vec2i32;
  typedef Vec2T<Int64> Vec2i64;


  typedef Vec2T<UInt> Vec2u;
  typedef Vec2T<UInt8> Vec2u8;
  typedef Vec2T<UInt16> Vec2u16;
  typedef Vec2T<UInt32> Vec2u32;
  typedef Vec2T<UInt64> Vec2u64;


#ifndef SWIG
  const Vec2 X_AXIS_2D( 1.0, 0.0 );
  const Vec2 Y_AXIS_2D( 0.0, 1.0 );
#endif

  // Hash function
  template <typename T>
  struct HashFn< Vec2T<T> >
  {
    AGX_FORCE_INLINE agx::UInt32 operator()(const Vec2T<T>& id) const
    {
      const agx::UInt32 p1 = 73856093UL;
      const agx::UInt32 p2 = 19349663UL;

      return (p1 * (agx::UInt32)id[0]) ^ (p2 * (agx::UInt32)id[1]);
    }
  };

  AGX_FORCE_INLINE Vec2i calculateCellId(Real x, Real y, Real invCellSize)
  {
    agx::Vec2i cell;
    cell[0] = (Vec2i::Type)(x * invCellSize);
    cell[1] = (Vec2i::Type)(y * invCellSize);

    if (x < Real(0.0)) cell[0]--;
    if (y < Real(0.0)) cell[1]--;

    return cell;
  }

  AGX_FORCE_INLINE Vec2i calculateCellId(const Vec2& position, Real invCellSize)
  {
    return calculateCellId(position[0], position[1], invCellSize);
  }

}   // end of namespace agx


AGX_TYPE_BINDING(agx::Vec2f, "Vec2")
AGX_TYPE_BINDING(agx::Vec2d, "Vec2")

AGX_TYPE_BINDING(agx::Vec2i8 , "Vec2i")
AGX_TYPE_BINDING(agx::Vec2i16, "Vec2i")
AGX_TYPE_BINDING(agx::Vec2i32, "Vec2i")
AGX_TYPE_BINDING(agx::Vec2i64, "Vec2i")

AGX_TYPE_BINDING(agx::Vec2u8 , "Vec2u")
AGX_TYPE_BINDING(agx::Vec2u16, "Vec2u")
AGX_TYPE_BINDING(agx::Vec2u32, "Vec2u")
AGX_TYPE_BINDING(agx::Vec2u64, "Vec2u")


#endif
