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

#ifndef AGX_HASH_H
#define AGX_HASH_H

#include <agx/agxCore_export.h>
#include <agx/Integer.h>

namespace agx
{
  // Hash wrapper, do not provide template specialization for this
  template <typename T>
  AGXCORE_EXPORT UInt32 hash( const T& key );

  // Hash combinator
  UInt32 AGXCORE_EXPORT hash( UInt32 h1, UInt32 h2 );


  // Implement template specializations for supported hashable types
  // default implementation calls key.hash(), so the easiest way is to add a hash method to the class
  // for native types template specializations is the only option
  template <typename T>
  struct HashFn
  {
    UInt32 operator()(const T& key) const;
  };


  // Hash key comparison function
  template <typename T1, typename T2>
  bool hashKeyEqual(const T1& key1, const T2& key2);


  // Explicitly pack as agx::UInt64 instead of SymmetricPair<agx::UInt32> due to faster hashing
  AGXCORE_EXPORT agx::UInt64 buildHashKey(agx::UInt32 id1, agx::UInt32 id2);
  AGXCORE_EXPORT bool hashKeyContains(agx::UInt64 key, agx::UInt32 id);
  AGXCORE_EXPORT void splitHashKey(agx::UInt64 key, agx::UInt32& id1, agx::UInt32& id2);
}


#endif /* AGX_HASH_H */
