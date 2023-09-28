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

#ifndef AGX_STACKARRAY_H
#define AGX_STACKARRAY_H

#include <agx/agx.h>
#include <agx/debug.h>
#include <array>
#include <cstring>
#include <initializer_list>
#include <ostream>

namespace agx
{
  /**
  * Templated stack array class. Should be used for POD classes/structs only.
  */
  template <typename T, size_t N >
  class StackArray
  {

  public:
    StackArray();

    StackArray(size_t size, const T& defaultValue = T());

    StackArray(std::initializer_list<T> values);

    /**
    Get the size of the array (number of filled slots).
    */
    size_t size() const;

    /// Is the StackArray empty?
    bool empty() const;

    T& operator[] (size_t i);

    const T& operator[] (size_t i) const;

    T& at(size_t i);

    const T& at(size_t i) const;

    void push_back(const T& value);

    /*
    Removes top element. Does not call destructor for this element.
    Should not be used if size()==0.
    */
    void pop_back();

    /*
    Copy constructor with type conversion.
    \param rhs: Another StackArray of same type T and with capacity N2 <= N.
    \retval: A reference to the constructed StackArray.
    */
    template <size_t N2>
    StackArray<T,N>& operator= (const StackArray<T,N2>& rhs);

    /*
    Removes all elements. Does not call destructor for these elements.
    */
    void clear();

    T& back();

    const T& back() const;

    T& front();

    const T& front() const;

    T* begin() noexcept;
    const T* begin() const noexcept;

    T* end() noexcept;
    const T* end() const noexcept;

    void erase(size_t index);

  private:
    template <typename T2, size_t N2>
    friend std::ostream& operator << ( std::ostream& output, const StackArray<T2, N2>& array);


  private:
    std::array<T, N> m_elements;
    size_t m_size;
  };

  template <typename T, size_t N>
  T* begin(StackArray<T, N>& array);

  template <typename T, size_t N>
  const T* begin(const StackArray<T, N>& array);

  template <typename T, size_t N>
  T* end(StackArray<T, N>& array);

  template <typename T, size_t N>
  const T* end(const StackArray<T, N>& array);

  /// Implementations
  template <typename T, size_t N>
  AGX_FORCE_INLINE StackArray<T, N>::StackArray() : m_elements(), m_size(0)
  {
    m_elements.fill( T() );
  }

  template <typename T, size_t N>
  template <size_t N2>
  AGX_FORCE_INLINE StackArray<T,N>& StackArray<T, N>::operator= (const StackArray<T,N2>& rhs)
  {
    agxAssert(rhs.size() <= N);
    m_size = rhs.size();
    for (size_t i = 0; i < m_size; ++i)
      m_elements[i] = rhs[i];
    return *this;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE StackArray<T, N>::StackArray(size_t size, const T& defaultValue /*= T()*/) : m_size(size)
  {
    m_elements.fill( T() );
    agxAssert(m_size <= N);
    for (size_t i = 0; i < size; ++i)
      m_elements[i] = defaultValue;
  }


  template <typename T, size_t N>
  inline StackArray<T, N>::StackArray(std::initializer_list<T> values)
  {
    agxAssert(values.size() <= N);
    std::copy(std::begin(values), std::end(values), std::begin(m_elements));
    m_size = values.size();
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE size_t StackArray<T, N>::size() const
  {
    return m_size;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE bool StackArray<T, N>::empty() const
  {
    return m_size == 0;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE T& StackArray<T, N>::operator[] (size_t i)
  {
    agxAssert(i < m_size);
    return m_elements[i];
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE const T& StackArray<T, N>::operator[] (size_t i) const
  {
    agxAssert(i < m_size);
    return m_elements[i];
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE T& StackArray<T, N>::at(size_t i)
  {
    agxAssert(i < m_size);
    return m_elements[i];
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE const T& StackArray<T, N>::at(size_t i) const
  {
    agxAssert(i < m_size);
    return m_elements[i];
  }

  template <typename T, size_t N>
  AGX_FORCE_INLINE void StackArray<T, N>::push_back(const T& value)
  {
    agxAssert(m_size < N);
    m_elements[m_size++] = value;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE void StackArray<T, N>::pop_back()
  {
    agxAssert(m_size > 0);
    m_size--;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE void StackArray<T, N>::clear()
  {
    m_size = 0;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE T& StackArray<T, N>::back()
  {
    agxAssert(m_size > 0);
    return m_elements[m_size - 1];
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE const T& StackArray<T, N>::back() const
  {
    agxAssert(m_size > 0);
    return m_elements[m_size - 1];
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE T& StackArray<T, N>::front()
  {
    agxAssert(m_size > 0);
    return m_elements[0];
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE const T& StackArray<T, N>::front() const
  {
    agxAssert(m_size > 0);
    return m_elements[0];
  }


  template <typename T, size_t N>
  T* StackArray<T, N>::begin() noexcept
  {
    return &m_elements[0];
  }

  template <typename T, size_t N>
  const T* StackArray<T, N>::begin() const noexcept
  {
    return &m_elements[0];
  }

  template <typename T, size_t N>
  T* StackArray<T, N>::end() noexcept
  {
    return begin() + size();
  }

  template <typename T, size_t N>
  const T* StackArray<T, N>::end() const noexcept
  {
    return begin() + size();
  }

  template <typename T, size_t N>
  AGX_FORCE_INLINE std::ostream& operator << ( std::ostream& output, const StackArray<T, N>& array)
  {
    for (size_t i = 0; i < array.size(); i++) {
      output << array[i];
      if (i < array.size()-1)
        output << ", ";
    }
    return output;
  }


  template <typename T, size_t N>
  AGX_FORCE_INLINE void StackArray<T, N>::erase(size_t index)
  {
    agxAssert( index < m_size);
    for (size_t i = index + 1; i < m_size; i++)
      m_elements[i-1] = m_elements[i];

    m_size--;
  }


  template <typename T, size_t N>
  T* begin(StackArray<T, N>& array)
  {
    return array.begin();
  }

  template <typename T, size_t N>
  const T* begin(const StackArray<T, N>& array)
  {
    return array.begin();
  }

  template <typename T, size_t N>
  T* end(StackArray<T, N>& array)
  {
    return array.end();
  }

  template <typename T, size_t N>
  const T* end(const StackArray<T, N>& array)
  {
    return array.end();
  }

}




#endif /* AGX_STACKARRAY_H */
