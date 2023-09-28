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


#ifndef AGXDATA_HASH_SET_H
#define AGXDATA_HASH_SET_H

#if 0
#include <agx/HashSet.h>
#include <agxData/ArrayAllocator.h>

namespace agxData
{
  template <typename KeyT, typename HashT = agx::HashFn<KeyT>, typename Allocator = ArrayAllocator::Proxy>
  class HashSet : public agx::HashSet<KeyT, HashT, Allocator>
  {
  public:
    AGX_FORCE_INLINE HashSet(const Allocator& allocator = Allocator()) : agx::HashSet<KeyT, HashT, Allocator>(allocator)
    {}

    AGX_FORCE_INLINE HashSet& operator=(const HashSet<KeyT, HashT, Allocator>& other)
    {
      agx::HashSet<KeyT, HashT, Allocator>::operator=(other);
      return *this;
    }
  };

  typedef HashSet<agx::UInt32> GenericHashSet;
}
AGX_TEMPLATED_TYPE_BINDING(agxData::HashSet, "HashSet")
#endif

#endif /* AGXDATA_HASH_SET_H */
