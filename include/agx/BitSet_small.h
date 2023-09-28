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

#ifndef AGX_BITSET_SMALL_H
#define AGX_BITSET_SMALL_H

#include <agx/macros.h>

namespace agx
{
  /**
  Special class for small bitset, because minimum bitset size in clang/gcc is 8 bytes, and we only need 4.
  */
  template <typename T>
  class BitSet_small
  {
  public:
    BitSet_small(T state = T(0));
    ~BitSet_small();

    bool test(size_t bit) const;
    void set(size_t bit, bool flag);

    T get() const;
    T state() const;

    T& get();
    T& state();

  protected:
    T m_state;
  };


  ////////////////////////////////////////////////////
  template <typename T>
  AGX_FORCE_INLINE BitSet_small<T>::BitSet_small(T state) : m_state(state)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE BitSet_small<T>::~BitSet_small()
  {
  }

  template <typename T>
  AGX_FORCE_INLINE bool BitSet_small<T>::test(size_t bit) const
  {
    const T mask = (T)(1 << bit);
    return (m_state & mask) != 0;
  }

  template <typename T>
  AGX_FORCE_INLINE void BitSet_small<T>::set(size_t bit, bool flag)
  {
    const T mask = (T)(1 << bit);
    m_state = (m_state & ~mask) | ((T)flag << bit);
  }

  template <typename T>
  AGX_FORCE_INLINE T BitSet_small<T>::get() const { return m_state; }

  template <typename T>
  AGX_FORCE_INLINE T BitSet_small<T>::state() const { return m_state; }

  template <typename T>
  AGX_FORCE_INLINE T& BitSet_small<T>::get() { return m_state; }

  template <typename T>
  AGX_FORCE_INLINE T& BitSet_small<T>::state() { return m_state; }

}


#endif /* AGX_BITSET_SMALL_H */
