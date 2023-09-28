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

#ifndef AGX_SYMMETRICPAIR_H
#define AGX_SYMMETRICPAIR_H

#include <utility>
#include <agx/macros.h>
#include <agx/hash.h>

namespace agx
{

  /**
  A std::pair, with both elements of the same type, and symmetric so (a, b) == (b, a)
  */
  template<typename T>
  class SymmetricPair : public std::pair<T, T>
  {
    public:
      SymmetricPair() : std::pair<T, T>() {}
      SymmetricPair(const T& a, const T& b) : std::pair<T, T>(a, b) {}

      AGX_FORCE_INLINE bool contains (const T& a ) const
      {
        return ( a == this->first || a == this->second );
      }

      // Hash function
      UInt32 hash() const;
  };

  template<typename T>
  AGX_FORCE_INLINE bool operator==(const SymmetricPair<T>& a, const SymmetricPair<T>& b)
  {
    return (a.first == b.first && a.second == b.second) || (a.first == b.second && a.second == b.first);
  }

  template<typename T>
  AGX_FORCE_INLINE bool operator!=(const SymmetricPair<T>& a, const SymmetricPair<T>& b)
  {
    return !(a == b);
  }

  template<typename T>
  AGX_FORCE_INLINE UInt32 SymmetricPair<T>::hash() const
  {
    UInt32 h1 = agx::hash(this->first);
    UInt32 h2 = agx::hash(this->second);

    if (h1 > h2 )
      std::swap(h1, h2);

    return agx::hash( h1, h2 );
  }

  template<typename T>
  std::ostream& operator << ( std::ostream& output, const SymmetricPair<T>& pair)
  {
    output << pair.first << " : " << pair.second;
    return output;
  }
}

#endif /* AGX_SYMMETRICPAIR_H */
