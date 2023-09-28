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

#ifndef AGXDATA_INTEGER_H
#define AGXDATA_INTEGER_H

#include <agx/config.h>
#include <agx/stdint.h>
#include <stdlib.h>
#include <limits>

namespace agx
{

#if AGX_64BIT_ARCHITECTURE
  #define AGX_LONG_IS_DEFAULT true
  typedef uint64_t UInt;
  typedef int64_t Int;
#else
  #define AGX_LONG_IS_DEFAULT false
  typedef uint32_t UInt;
  typedef int32_t Int;
#endif

  typedef uint8_t UInt8;
  typedef uint16_t UInt16;
  typedef uint32_t UInt32;
  typedef uint64_t UInt64;

  typedef int8_t Int8;
  typedef int16_t Int16;
  typedef int32_t Int32;
  typedef int64_t Int64;

  typedef bool Bool;

  typedef size_t Size;

  typedef UInt32 Index;

  // Use UInt types for non-default index types
  // typedef UInt8 Index8;
  // typedef UInt16 Index16;
  // typedef UInt32 Index32;
  // typedef UInt64 Index64;

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

  // Name clash with global macro definitions
  const UInt UIntMax = std::numeric_limits<UInt>::max();
  const UInt8 UInt8Max = std::numeric_limits<UInt8>::max();
  const UInt16 UInt16Max = std::numeric_limits<UInt16>::max();
  const UInt32 UInt32Max = std::numeric_limits<UInt32>::max();
  const UInt64 UInt64Max = std::numeric_limits<UInt64>::max();

  const Int IntMax = std::numeric_limits<Int>::max();
  const Int8 Int8Max = std::numeric_limits<Int8>::max();
  const Int16 Int16Max = std::numeric_limits<Int16>::max();
  const Int32 Int32Max = std::numeric_limits<Int32>::max();
  const Int64 Int64Max = std::numeric_limits<Int64>::max();

  const Size SizeMax = std::numeric_limits<Size>::max();
}

#endif /* _AGXDATA_INTEGER_H_ */
