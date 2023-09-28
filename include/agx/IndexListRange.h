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

#ifndef AGX_INDEXLISTRANGE_H
#define AGX_INDEXLISTRANGE_H

#include <agxData/Array.h>

namespace agx
{
  template< typename T>
  class IndexListRange
  {
  public:
    typedef T Type;
    typedef const T* iterator;
  public:
    IndexListRange( const agxData::Array<T>& indexList, IndexRange range );

    iterator begin() const;
    iterator end() const;

    size_t size() const;

    const T& operator[]( size_t index ) const;

  private:
    const agxData::Array<T>& m_indexList;

    IndexListRange<T>& operator=(const IndexListRange<T>&) { agxAbort(); }
    IndexRange m_range;
  };


  template< typename T>
  IndexListRange<T>::IndexListRange( const agxData::Array<T>& indexList, IndexRange range ) :
    m_indexList( indexList ) , m_range( range )
  {
    agxAssert( range.end() <= indexList.size() );
  }

  template< typename T>
  typename IndexListRange<T>::iterator IndexListRange<T>::begin() const
  {
    return &m_indexList[ m_range.begin() ];
  }

  template< typename T>
  typename IndexListRange<T>::iterator IndexListRange<T>::end() const
  {
    // Special handling when the range extends to the end of the Array since
    // we can't take the address of the 'end' element.
    return  m_range.end() == m_indexList.size() ? m_indexList.end()  :  &m_indexList[ m_range.end() ];
  }

  template< typename T>
  size_t IndexListRange<T>::size() const
  {
    return m_range.size();
  }

  template< typename T>
  const T& IndexListRange<T>::operator[] ( size_t index ) const
  {
    return m_indexList[ m_range[index] ];
  }
}



#endif
