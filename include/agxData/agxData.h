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

#ifndef AGXDATA_AGXDATA_H
#define AGXDATA_AGXDATA_H

#include <agx/String.h>
#include <agx/Integer.h>

/**
\namespace agxData
\brief Contains classes for low level data storage for AGX.
*/
namespace agxData
{
  template <typename T>
  class Array;

  /**
  Data access mode.
  */
  enum AccessMode
  {
    READ = 0x1,
    WRITE = 0x2,
    READ_WRITE = READ | WRITE
  };

  AGXCORE_EXPORT agx::UInt getAccessMode(const agx::String& modeName);
  AGXCORE_EXPORT agx::String getAccessModeString(agx::UInt mode);

  typedef agx::UInt8 Byte;
  typedef Byte* BytePtr;
  typedef const Byte* ConstBytePtr;
  typedef agx::VectorPOD<Byte> ByteVector;

  typedef Array<agx::Index> IndexArray;

}


#endif /* _AGXDATA_AGXDATA_H_ */
