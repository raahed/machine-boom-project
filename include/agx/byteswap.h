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

#ifndef AGX_BYTESWAP_H
#define AGX_BYTESWAP_H

#include <agx/agx.h>

#ifdef _MSC_VER
#else
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#else //LINUX
// The byteswap_.* family of macros inject code containing the deprecated 'register' keyword.
#include <agx/PushDisableWarnings.h>
#include <byteswap.h>
#endif
#endif

namespace agx
{
#ifdef _MSC_VER
  static inline uint16_t ByteSwap16 ( uint16_t x )
  {
    return ( x >> 8 ) | ( x << 8 );
  }

  static inline uint32_t ByteSwap32 ( uint32_t x )
  {
    return ( ByteSwap16 ( x & 0xffff ) << 16 ) | ( ByteSwap16 ( x >> 16 ) );
  }

  static inline uint64_t ByteSwap64 ( uint64_t x )
  {
    return ( ( ( uint64_t ) ByteSwap32 ( uint32_t( x & 0xffffffffull ) ) << 32 ) ) |
           ( ByteSwap32 ( uint32_t( x >> 32 ) ) );
  }

#else
#ifdef __APPLE__
  static inline uint16_t ByteSwap16 ( uint16_t x )
  {
    return CFSwapInt16(x);
  }

  static inline uint32_t ByteSwap32 ( uint32_t x )
  {
    return CFSwapInt32(x);
  }

  static inline uint64_t ByteSwap64 ( uint64_t x )
  {
    return CFSwapInt64(x);
  }

#else //LINUX

  static inline uint16_t ByteSwap16 ( uint16_t x )
  {
    return bswap_16(x);
  }

  static inline uint32_t ByteSwap32 ( uint32_t x )
  {
    return bswap_32(x);
  }

  static inline uint64_t ByteSwap64 ( uint64_t x )
  {
    return bswap_64(x);
  }

#endif
#endif
}

#ifdef __linux__
#include <agx/PopDisableWarnings.h>
#endif


#endif /* _AGX_BYTESWAP_H_ */
