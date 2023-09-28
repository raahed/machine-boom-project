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

#ifndef AGX_PACK_H
#define AGX_PACK_H

namespace agx
{

  union uint32_t_union {
    struct uint32_t_struct {
      uint32_t byte3:8,byte2:8,byte1:8,byte0:8;
    };
    uint32_t_struct bytes;
    uint32_t integer;
  };

  /**
  Pack 4 8 bit unsigned integers into one 32bit unsigned integer
  \return a 32bit unsigned integer with 4 bytes packed into it
  */
  inline uint32_t PACK_UINT32(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3)
  {
    uint32_t_union v;
    v.integer=0;

    v.bytes.byte0 = b0;
    v.bytes.byte1 = b1;
    v.bytes.byte2 = b2;
    v.bytes.byte3 = b3;

    return v.integer;
  }


  /**
  Unpack 4 8 bit unsigned integers from one unsigned 32bit integer
  */
  inline void UNPACK_UINT32(uint32_t val, uint8_t& c0, uint8_t& c1, uint8_t& c2, uint8_t& c3)
  {
    uint32_t_union v;
    v.integer = val;
    c0 = v.bytes.byte0;
    c1 = v.bytes.byte1;
    c2 = v.bytes.byte2;
    c3 = v.bytes.byte3;
  }
}
#endif

