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

#ifndef AGXDATA_RANGE_H
#define AGXDATA_RANGE_H

#include <agx/agx.h>
#include <agx/Integer.h>
#include <agx/Vector.h>
#include <iosfwd>

namespace agx
{

  /**
  A range of indices.
  */
  template <typename T>
  class IndexRangeT
  {
  public:
    class iterator;
    typedef T Type;

    IndexRangeT();
    IndexRangeT(T start, T end);

    template <typename T2>
    operator IndexRangeT<T2>() const;

    /**
    \return The beginning of the range.
    */
    T& begin();
    const T& begin() const;

    /**
    \return The end of the range.
    */
    T& end();
    const T& end() const;

    /**
    []-operator
    */
    T operator[] (T localIndex) const;

    /**
    Set the range.
    */
    void set(T begin, T end);


    /**
    \return The size of the range.
    */
    T size() const;

    /**
    \return True if the range is empty.
    */
    bool empty() const;

    /**
    \return True if the range contains a specified index.
    */
    bool contains(T index) const;

    /**
    \return True if the range has overlap with another range.
    */
    bool hasOverlap(const IndexRangeT<T>& other) const;

    /**
    Clamp range using another range as limits.
    */
    void clamp(const IndexRangeT<T>& other);

    /**
    \return false if begin > end
    */
    bool valid() const;


    bool operator== (const IndexRangeT<T>& other) const;
    bool operator!= (const IndexRangeT<T>& other) const;

    // TODO Unclear semantics?
    IndexRangeT<T> operator + ( T rhs ) const;
    IndexRangeT<T>& operator += ( T rhs );
    IndexRangeT<T> operator - ( T rhs ) const;
    IndexRangeT<T>& operator -= ( T rhs );


  private:
    T m_start;
    T m_end;
  };

  template <typename T>
  IndexRangeT<T> makeIndexRange(const T begin, const T end);

  /**
  Index range iterator.
  */
  template <typename T>
  class IndexRangeT<T>::iterator
  {
  public:
    iterator();
    iterator(T index);

    iterator& operator++();
    iterator operator++(int);

    operator T() const;

    T operator*() const;

  private:
    T m_index;
  };




  typedef IndexRangeT<agx::UInt> IndexRange;
  typedef IndexRangeT<agx::UInt32> IndexRange32;
  typedef IndexRangeT<agx::UInt64> IndexRange64;

  typedef VectorPOD<IndexRange> IndexRangeVector;
  typedef VectorPOD<IndexRange32> IndexRange32Vector;
  typedef VectorPOD<IndexRange64> IndexRange64Vector;


  /* Implementation */
  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>::IndexRangeT() : m_start(0), m_end(0) {}

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>::IndexRangeT(T start, T end) : m_start(start), m_end(end) { agxAssert(end >= start); }

  template <typename T> template <typename T2>
  AGX_FORCE_INLINE IndexRangeT<T>::operator IndexRangeT<T2>() const
  {
    return IndexRangeT<T2>(T2(m_start), T2(m_end));
  }


  template <typename T>
  AGX_FORCE_INLINE void IndexRangeT<T>::set(T begin, T end) { agxAssert(end >= begin); m_start = begin; m_end = end; }


  template <typename T>
  AGX_FORCE_INLINE T IndexRangeT<T>::size() const { return m_end - m_start; }

  template <typename T>
  AGX_FORCE_INLINE bool IndexRangeT<T>::empty() const { return m_end == m_start; }

  template <typename T>
  AGX_FORCE_INLINE bool IndexRangeT<T>::contains(T index) const { return index >= m_start && index < m_end; }

  template <typename T>
  AGX_FORCE_INLINE bool IndexRangeT<T>::hasOverlap(const IndexRangeT<T>&) const
  {
    agxAbort();
    return false;
  }

  template <>
  AGX_FORCE_INLINE bool IndexRangeT<agx::UInt64>::hasOverlap(const IndexRangeT<agx::UInt64>& other) const
  {
    agx::UInt64 a = m_start - other.m_end;
    agx::UInt64 b = m_end - other.m_start;
    return a != 0 && b != 0 && (a >> 63) != (b >> 63);
  }

  template <>
  AGX_FORCE_INLINE bool IndexRangeT<agx::UInt32>::hasOverlap(const IndexRangeT<agx::UInt32>& other) const
  {
    agx::UInt32 a = m_start - other.m_end;
    agx::UInt32 b = m_end - other.m_start;
    return a != 0 && b != 0 && (a >> 31) != (b >> 31);
  }

  template <typename T>
  AGX_FORCE_INLINE void IndexRangeT<T>::clamp(const IndexRangeT<T>& other)
  {
    if (m_start < other.m_start) m_start = other.m_start;
    if (m_end > other.m_end) m_end = other.m_end;
  }

  template <typename T>
  AGX_FORCE_INLINE bool IndexRangeT<T>::valid() const
  {
    return m_start <= m_end;
  }


  template <typename T>
  AGX_FORCE_INLINE T& IndexRangeT<T>::begin() { return m_start; }

  template <typename T>
  AGX_FORCE_INLINE T& IndexRangeT<T>::end() { return m_end; }


  template <typename T>
  AGX_FORCE_INLINE const T& IndexRangeT<T>::begin() const { return m_start; }

  template <typename T>
  AGX_FORCE_INLINE const T& IndexRangeT<T>::end() const { return m_end; }

  template <typename T>
  AGX_FORCE_INLINE T IndexRangeT<T>::operator[] (T localIndex) const { return m_start + localIndex; }


  template <typename T>
  AGX_FORCE_INLINE bool IndexRangeT<T>::operator== (const IndexRangeT<T>& other) const
  {
    return m_start == other.m_start && m_end == other.m_end;
  }

  template <typename T>
  AGX_FORCE_INLINE bool IndexRangeT<T>::operator!= (const IndexRangeT<T>& other) const
  {
    return m_start != other.m_start || m_end != other.m_end;
  }


  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T> IndexRangeT<T>::operator + ( T rhs ) const { return IndexRangeT(m_start+rhs, m_end+rhs); }

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>& IndexRangeT<T>::operator += ( T rhs ) { m_start += rhs; m_end += rhs; return *this; }

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T> IndexRangeT<T>::operator - ( T rhs ) const { return IndexRangeT(m_start-rhs, m_end-rhs); }

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>& IndexRangeT<T>::operator -= ( T rhs ) { m_start -= rhs; m_end -= rhs; return *this; }



  template <typename T>
  std::ostream& operator << ( std::ostream& output, const IndexRangeT<T>& range )
  {
    output << (T)range.begin() << ":" << (T)range.end();
    return output;
  }


  // This function won't be necessary once we get class template argument
  // deduction with C++17.
  template <typename T>
  IndexRangeT<T> makeIndexRange(const T begin, const T end)
  {
    return IndexRangeT<T>(begin, end);
  }



  //// Iterator /////////////////////

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>::iterator::iterator() : m_index(agx::InvalidIndex)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>::iterator::iterator(T index) : m_index(index)
  {
  }

  template <typename T>
  AGX_FORCE_INLINE typename IndexRangeT<T>::iterator& IndexRangeT<T>::iterator::operator++()
  {
    ++m_index;
    return *this;
  }

  template <typename T>
  AGX_FORCE_INLINE typename IndexRangeT<T>::iterator IndexRangeT<T>::iterator::operator++(int)
  {
    ++m_index;
    return IndexRangeT<T>::iterator(m_index-1);
  }

  template <typename T>
  AGX_FORCE_INLINE IndexRangeT<T>::iterator::operator T() const { return m_index; }

  template <typename T>
  AGX_FORCE_INLINE T IndexRangeT<T>::iterator::operator*() const { return m_index; }


}


#endif /* _AGXDATA_RANGE_H_ */
