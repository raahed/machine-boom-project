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

#ifndef AGX_MORTON_H
#define AGX_MORTON_H

#include <agx/Math.h>
#include <agx/Vec3.h>

// Utility functions for calculating Z-order (Morton order) indices
// http://en.wikipedia.org/wiki/Z-order_curve

// ULL suffix necessary for 32bit build
#define AGX_LONG_CONSTANT(x) x ## ULL

namespace agx
{
  UInt32 morton_reference(UInt32 x, UInt32 y, UInt32 z);
  UInt64 morton_reference(UInt64 x, UInt64 y, UInt64 z);
  UInt32 morton_reference(UInt32 x, UInt32 y);
  UInt64 morton_reference(UInt64 x, UInt64 y);

  ////////////////////////////////////////////////////////////////////////////
  // 32 bit
  ////////////////////////////////////////////////////////////////////////////

  // From chapter 7.3.5 of Christer Ericson's book "Real time collision detection"
  // Is approximately 3x faster than naive loop implementation.
  AGX_FORCE_INLINE UInt32 morton_part1By2(UInt32 n)
  {
    n = n & 0x3FF; // Only keep low 10 bits
    n = (n | (n << 16)) & 0xFF0000FF;
    n = (n | (n <<  8)) & 0x0300F00F;
    n = (n | (n <<  4)) & 0x030C30C3;
    n = (n | (n <<  2)) & 0x09249249;
    return n;
  }


  AGX_FORCE_INLINE UInt32 morton(UInt32 x, UInt32 y, UInt32 z)
  {
    UInt32 result = (morton_part1By2(z) << 2) + (morton_part1By2(y) << 1) + morton_part1By2(x);

    #ifdef AGX_DEBUG
    agxAssert(result == morton_reference(x, y, z));
    #endif

    return result;
  }

  // Reference implementation
  AGX_FORCE_INLINE UInt32 morton_reference(UInt32 x, UInt32 y, UInt32 z)
  {
    UInt32 result = 0;

    for(size_t i = 0; i < 10; i++)
    {
      UInt32 mask = 1 << i;
      result |= (x & mask) << (i * 2) | (y & mask) << (i * 2 + 1) | (z & mask) << (i * 2 + 2);
    }

    return result;
  }


  // 2d
  AGX_FORCE_INLINE UInt32 morton_part1By1(UInt32 n)
  {
    n = n & 0xFFFF; // Only keep low 16 bits
    n = (n | (n <<  8)) & 0x00FF00FF;
    n = (n | (n <<  4)) & 0x0F0F0F0F;
    n = (n | (n <<  2)) & 0x33333333;
    n = (n | (n <<  1)) & 0x55555555;
    return n;
  }

  AGX_FORCE_INLINE UInt32 morton(UInt32 x, UInt32 y)
  {
    UInt32 result = (morton_part1By1(y) << 1) + morton_part1By1(x);

    #ifdef AGX_DEBUG
    agxAssert(result == morton_reference(x, y));
    #endif

    return result;
  }

  // Reference implementation
  AGX_FORCE_INLINE UInt32 morton_reference(UInt32 x, UInt32 y)
  {
    UInt32 result = 0;

    for(size_t i = 0; i < 16; i++)
    {
      UInt32 mask = 1 << i;
      result |= (x & mask) << i | (y & mask) << (i + 1);
    }

    return result;
  }


  // Convenience methods
  AGX_FORCE_INLINE UInt32 morton(const Vec3u32& vec) { return morton((UInt32)vec[0], (UInt32)vec[1], (UInt32)vec[2]); }
  AGX_FORCE_INLINE UInt32 morton(const Vec3i32& vec)
  {
    agxAssert(vec[0] >= 0 && vec[1] >= 0 && vec[2] >= 0);
    return morton((UInt32)vec[0], (UInt32)vec[1], (UInt32)vec[2]);
  }

  AGX_FORCE_INLINE UInt32 morton(const Vec2u32& vec) { return morton((UInt32)vec[0], (UInt32)vec[1]); }
  AGX_FORCE_INLINE UInt32 morton(const Vec2i32& vec)
  {
    agxAssert(vec[0] >= 0 && vec[1] >= 0);
    return morton((UInt32)vec[0], (UInt32)vec[1]);
  }


  ///////////////////////////////////////////////////////////////////////////////////
  // 64 bit
  ///////////////////////////////////////////////////////////////////////////////////


  // Based on chapter 7.3.5 of Christer Ericson's book "Real time collision detection"
  // At least 5x faster than naive loop implementation
  AGX_FORCE_INLINE UInt64 morton_part1By2(UInt64 n)
  {
    n = n & 0x1FFFFF; // Only keep low 21 bits
    n = (n | (n << 32)) & AGX_LONG_CONSTANT(0x001F00000000FFFF);
    n = (n | (n << 16)) & AGX_LONG_CONSTANT(0x001F0000FF0000FF);
    n = (n | (n <<  8)) & AGX_LONG_CONSTANT(0x100F00F00F00F00F);
    n = (n | (n <<  4)) & AGX_LONG_CONSTANT(0x10C30C30C30C30C3);
    n = (n | (n <<  2)) & AGX_LONG_CONSTANT(0x1249249249249249);
    return n;
  }

  AGX_FORCE_INLINE UInt64 morton(UInt64 x, UInt64 y, UInt64 z)
  {
    UInt64 result = (morton_part1By2(z) << 2) + (morton_part1By2(y) << 1) + morton_part1By2(x);

    #ifdef AGX_DEBUG
    agxAssert(result == morton_reference(x, y, z));
    #endif

    return result;
  }

  // Reference implementation
  AGX_FORCE_INLINE UInt64 morton_reference(UInt64 x, UInt64 y, UInt64 z)
  {
    UInt64 result = 0;

    for(UInt64 i = 0; i < 21; i++)
    {
      UInt64 mask = static_cast<UInt64>(1) << i;
      result |= (x & mask) << (i * 2) | (y & mask) << (i * 2 + 1) | (z & mask) << (i * 2 + 2);
    }

    return result;
  }

  // 2d
  AGX_FORCE_INLINE UInt64 morton_part1By1(UInt64 n)
  {
    n = n & 0xFFFFFFFF; // Only keep low 32 bits
    n = (n | (n <<  16)) & AGX_LONG_CONSTANT(0x0000FFFF0000FFFF);
    n = (n | (n <<   8)) & AGX_LONG_CONSTANT(0x00FF00FF00FF00FF);
    n = (n | (n <<   4)) & AGX_LONG_CONSTANT(0x0F0F0F0F0F0F0F0F);
    n = (n | (n <<   2)) & AGX_LONG_CONSTANT(0x3333333333333333);
    n = (n | (n <<   1)) & AGX_LONG_CONSTANT(0x5555555555555555);
    return n;
  }

  AGX_FORCE_INLINE UInt64 morton(UInt64 x, UInt64 y)
  {
    UInt64 result = (morton_part1By1(y) << 1) + morton_part1By1(x);

    #ifdef AGX_DEBUG
    agxAssert(result == morton_reference(x, y));
    #endif

    return result;
  }

  // Reference implementation
  AGX_FORCE_INLINE UInt64 morton_reference(UInt64 x, UInt64 y)
  {
    UInt64 result = 0;

    for(UInt64 i = 0; i < 32; i++)
    {
      UInt64 mask = static_cast<UInt64>(1) << i;
      result |= (x & mask) << i | (y & mask) << (i + 1);
    }

    return result;
  }


  // Convenience methods
  AGX_FORCE_INLINE UInt64 morton(const Vec3u64& vec) { return morton((UInt64)vec[0], (UInt64)vec[1], (UInt64)vec[2]); }
  AGX_FORCE_INLINE UInt64 morton(const Vec3i64& vec)
  {
    agxAssert(vec[0] >= 0 && vec[1] >= 0 && vec[2] >= 0);
    return morton((UInt64)vec[0], (UInt64)vec[1], (UInt64)vec[2]);
  }

  AGX_FORCE_INLINE UInt64 morton(const Vec2u64& vec) { return morton((UInt64)vec[0], (UInt64)vec[1]); }
  AGX_FORCE_INLINE UInt64 morton(const Vec2i64& vec)
  {
    agxAssert(vec[0] >= 0 && vec[1] >= 0);
    return morton((UInt64)vec[0], (UInt64)vec[1]);
  }

}

#undef AGX_LONG_CONSTANT

#endif /* _AGX_MORTON_H_ */
