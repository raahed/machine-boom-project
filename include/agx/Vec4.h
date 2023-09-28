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

#ifndef AGX_VEC4_H
#define AGX_VEC4_H

#include <agx/Vec4Template.h>
#include <agxData/Type.h>

namespace agx
{
  typedef Vec4T<Real> Vec4;
  typedef Vec4T<Real32> Vec4f;
  typedef Vec4T<Real64> Vec4d;


  typedef Vec4T<Int> Vec4i;
  typedef Vec4T<Int8> Vec4i8;
  typedef Vec4T<Int16> Vec4i16;
  typedef Vec4T<Int32> Vec4i32;
  typedef Vec4T<Int64> Vec4i64;


  typedef Vec4T<UInt> Vec4u;
  typedef Vec4T<UInt8> Vec4u8;
  typedef Vec4T<UInt16> Vec4u16;
  typedef Vec4T<UInt32> Vec4u32;
  typedef Vec4T<UInt64> Vec4u64;

}   // end of namespace agx


AGX_TYPE_BINDING(agx::Vec4f, "Vec4")
AGX_TYPE_BINDING(agx::Vec4d, "Vec4")

AGX_TYPE_BINDING(agx::Vec4i8 , "Vec4i")
AGX_TYPE_BINDING(agx::Vec4i16, "Vec4i")
AGX_TYPE_BINDING(agx::Vec4i32, "Vec4i")
AGX_TYPE_BINDING(agx::Vec4i64, "Vec4i")

AGX_TYPE_BINDING(agx::Vec4u8 , "Vec4u")
AGX_TYPE_BINDING(agx::Vec4u16, "Vec4u")
AGX_TYPE_BINDING(agx::Vec4u32, "Vec4u")
AGX_TYPE_BINDING(agx::Vec4u64, "Vec4u")

#endif
