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

#ifndef AGXDATA_STACKARRAY_H
#define AGXDATA_STACKARRAY_H

#include <agxData/Array.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning (disable: 4324) // : 'agxData::StackArray<T,N>' : structure was padded due to __declspec(align()))
#endif

namespace agxData
{
  template <typename T, size_t N>
  class StackArray : public Array<T>
  {

  public:
    StackArray(const T& defaultValue = T());

  private:
    T m_elements[N];
  };



  /* Implementation */
  template <typename T, size_t N>
  AGX_FORCE_INLINE StackArray<T, N>::StackArray(const T& defaultValue)
  {
    this->m_ptr = m_elements;
    this->m_buffer = nullptr;
    this->m_range = agx::IndexRange32(0, N);

    for (size_t i = 0; i < N; ++i)
      m_elements[i] = defaultValue;
  }

  #if 0
  template <typename T>
  class Array2 : public StackArray<T, 2>
  {
  public:
    Array2(const T& val0, const T& val1);
  };

  template <typename T>
  AGX_FORCE_INLINE Array2<T>::Array2(const T& val0, const T& val1)
  {
    (*this)[0] = val0;
    (*this)[1] = val1;
  }
  #endif

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif /* AGXDATA_STACKARRAY_H */
