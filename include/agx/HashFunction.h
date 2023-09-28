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

#ifndef AGX_HASHFUNCTIONS_H
#define AGX_HASHFUNCTIONS_H

#include <stddef.h>
#include <cstring>

#include <agx/agx.h>
#include <agx/Integer.h>
#include <agx/String.h>
#include <agx/hash.h>

namespace agx
{

  /////////////////////////////////////////////////////////
  // See hash function declarations in agx/hash.h
  /////////////////////////////////////////////////////////




  // Implementation of basic hash functions


  AGX_FORCE_INLINE UInt64 buildHashKey(UInt32 id1, UInt32 id2)
  {
    return (id1 < id2) ? (((UInt64)id1 << 32) | (UInt64)id2) : (((UInt64)id2 << 32) | (UInt64)id1);
  }

  AGX_FORCE_INLINE bool hashKeyContains(UInt64 key, UInt32 id)
  {
    return (UInt32)key == id || (UInt32)(key >> 32) == id;
  }

  AGX_FORCE_INLINE void splitHashKey(UInt64 key, UInt32& id1, UInt32& id2)
  {
    id1 = (UInt32)key;
    id2 = (UInt32)(key >> 32);
  }

  template <typename T>
  UInt32 HashFn<T>::operator()(const T& key) const
  {
    return key.hash();
  }


  template <typename T>
  AGX_FORCE_INLINE UInt32 hash(const T& key)
  {
    HashFn<T> hashFn;
    return hashFn(key);
  }

  AGX_FORCE_INLINE UInt32 hash( UInt32 h1, UInt32 h2 )
  {
    return h1 ^ (h2 + 0x9e3779b9 + (h1<<6) + (h1>>2));
  }

  template <typename T1, typename T2>
  AGX_FORCE_INLINE bool hashKeyEqual(const T1& key1, const T2& key2)
  {
    return key1 == key2;
  }


  // Pair hashing
  template<typename T1, typename T2>
  struct HashFn< std::pair<T1, T2> >
  {
    AGX_FORCE_INLINE UInt32 operator()(const std::pair<T1,T2>& key) const
    {
      return hash(hash(key.first), hash(key.second));
    }
  };



  // Integer hash functions
  template <>
  struct HashFn<UInt32>
  {
    AGX_FORCE_INLINE UInt32 operator()(UInt32 key) const
    {
      /* http://www.concentric.net/~Ttwang/tech/inthash.htm */
      key = ~key + (key << 15); // key = (key << 15) - key - 1;
      key = key ^ (key >> 12);
      key = key + (key << 2);
      key = key ^ (key >> 4);
      key = key * 2057; // key = (key + (key << 3)) + (key << 11);
      key = key ^ (key >> 16);
      return key;
    }
  };

  template <>
  struct HashFn<Int32>
  {
    AGX_FORCE_INLINE UInt32 operator()(Int32 key) const
    {
      return hash((UInt32)key);
    }

  };

  template <>
  struct HashFn<UInt64>
  {
    AGX_FORCE_INLINE UInt32 operator()(UInt64 key) const
    {
      /* http://www.concentric.net/~Ttwang/tech/inthash.htm */
      key = (~key) + (key << 18); // key = (key << 18) - key - 1;
      key = key ^ (key >> 31);
      key = key * 21; // key = (key + (key << 2)) + (key << 4);
      key = key ^ (key >> 11);
      key = key + (key << 6);
      key = key ^ (key >> 22);
      return (UInt32)key;
    }
  };

  template <>
  struct HashFn<Int64>
  {
    AGX_FORCE_INLINE UInt32 operator()(Int64 key) const
    {
      return hash((UInt64)key);
    }

  };


#ifdef __APPLE__
  template <>
  struct HashFn<size_t>
  {
    AGX_FORCE_INLINE UInt32 operator()(size_t key) const
    {
    #if AGX_64BIT_ARCHITECTURE
      return hash((UInt64)(key));
    #else
      return hash((UInt32)(key));
    #endif
    }

  };
#endif

  // Pointer hashing

  template <typename T>
  struct HashFn<T *>
  {
    typedef const T *PtrT;

    AGX_FORCE_INLINE UInt32 operator()(const PtrT key) const
    {
    #if AGX_64BIT_ARCHITECTURE
      return hash((UInt64)(key));
    #else
      return hash((UInt32)(key));
    #endif
    }
  };

  // String hashing

  // Hash functions

  template <typename T>
  AGX_FORCE_INLINE UInt32 stringHash(const T& key, UInt32 startValue = 0)
  {
    UInt32 hashValue = startValue;

    for (UInt32 i = 0; i < key.length(); i++)
      hashValue = 37 * hashValue + (UInt32)key[i];

    return hashValue;
  }


  template<>
  struct HashFn<std::string>
  {
    AGX_FORCE_INLINE UInt32 operator()(const std::string& key) const
    {
      return agx::stringHash(key);
    }
  };


  template<>
  struct HashFn<agx::String>
  {
    AGX_FORCE_INLINE UInt32 operator()(const agx::String& key) const
    {
      return agx::stringHash(key);
    }
  };


  template<>
  struct HashFn<const char *>
  {
    AGX_FORCE_INLINE UInt32 operator()(const char *key) const
    {
      UInt32 hashValue = 0;

      while (*key)
        hashValue = 37 * hashValue + (UInt32)(*key++);

      return hashValue;
    }
  };


  AGX_FORCE_INLINE bool hashKeyEqual(char *key1, char *key2)
  {
    return strcmp(key1, key2) == 0;
  }

  AGX_FORCE_INLINE bool hashKeyEqual(const char *key1, const char *key2)
  {
    return strcmp(key1, key2) == 0;
  }


}

#endif /* _AGX_HASHFUNCTIONS_H_ */
