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

#pragma once

#include <agxSDK/LinkedSegmentIterator.h>

namespace agxSDK
{
  /**
  Segment range object for linked structure.
  */
  template<typename T>
  class LinkedSegmentRange
  {
    public:
      /**
      Create an empty range. Both \p begin and \p end will return iterators for
      which \p isEnd returns true.
      */
      LinkedSegmentRange();

      /**
      Construct given begin and end iterator.
      \param begin - begin iterator
      \param end - end iterator
      */
      LinkedSegmentRange( agxSDK::LinkedSegmentIterator<T> begin, agxSDK::LinkedSegmentIterator<T> end );

      /**
      Construct given begin segment and number of segments in the range.
      \param begin - begin iterator
      \param numSegments - number of segments in this range
      */
      LinkedSegmentRange( agxSDK::LinkedSegmentIterator<T> begin, agx::UInt numSegments );

      ~LinkedSegmentRange();


      agxSDK::LinkedSegmentRange<T>& operator=(const agxSDK::LinkedSegmentRange<T>& other);

      /**
      \return the begin iterator
      */
      agxSDK::LinkedSegmentIterator<T> begin() const;

      /**
      \return the end iterator
      */
      agxSDK::LinkedSegmentIterator<T> end() const;

      /**
      \return the first segment in this range
      */
      T* front() const;

      /**
      \return the last segment in this range
      */
      T* back() const;

      /**
      \return the segment with local index \p rangeIndex in this range (rangeIndex < size())
      */
      T* operator[]( agx::UInt rangeIndex ) const;

      /**
      \return the size of this range
      */
      agx::UInt size() const;

      /**
      \return true if this range is empty - otherwise false
      */
      agx::Bool empty() const;

    private:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Finds size of this range and updates boundaryless state of begin and end
      given their indices.
      */
      void updateSizeAndState();
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      LinkedSegmentIterator<T> m_begin;
      LinkedSegmentIterator<T> m_end;
      agx::UInt m_size;
  };




  template<typename T>
  LinkedSegmentRange<T>::LinkedSegmentRange()
      : m_begin()
      , m_end()
      , m_size(0)
  {
  }



  template<typename T>
  inline LinkedSegmentRange<T>::LinkedSegmentRange( LinkedSegmentIterator<T> begin, LinkedSegmentIterator<T> end )
    : m_begin( begin ), m_end( end ), m_size( 0u )
  {
    updateSizeAndState();
  }



  template<typename T>
  inline LinkedSegmentRange<T>::LinkedSegmentRange( LinkedSegmentIterator<T> begin, agx::UInt numSegments )
    : m_begin( begin ), m_end( m_begin.m_index, false, *m_begin.m_segments ), m_size( 0u )
  {
    // It's undefined, but possible, to create a range from begin to 10 * container.size().
    // This assert alerts the user something's probably wrong in their algorithm.
    // Note: numSegments == container.size() is valid since it can be the range from
    //       begin to end.
    agxAssert( numSegments <= m_begin.m_segments->size() );

    m_end.m_index += numSegments;
    if ( m_end.m_index > m_end.m_segments->size() )
      m_end.m_index = m_end.modSize( m_end.m_index, *m_end.m_segments );

    updateSizeAndState();
  }



  template<typename T>
  inline LinkedSegmentRange<T>::~LinkedSegmentRange()
  {
  }



  template<typename T>
  LinkedSegmentRange<T>& LinkedSegmentRange<T>::operator=(const LinkedSegmentRange<T>& other)
  {
    this->m_begin = other.m_begin;
    this->m_end = other.m_end;
    this->m_size = other.m_size;

    return *this;
  }



  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentRange<T>::begin() const
  {
    return m_begin;
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentRange<T>::end() const
  {
    return m_end;
  }

  template<typename T>
  inline T* LinkedSegmentRange<T>::front() const
  {
    if ( empty() )
      return nullptr;

    return *m_begin;
  }

  template<typename T>
  inline T* LinkedSegmentRange<T>::back() const
  {
    if ( empty() )
      return nullptr;

    return *( --LinkedSegmentIterator<T>( m_end ) );
  }

  template<typename T>
  inline T* LinkedSegmentRange<T>::operator[]( agx::UInt rangeIndex ) const
  {
    agxAssert( rangeIndex < m_size );
    return LinkedSegmentIterator<T>::incremented( m_begin, rangeIndex ).get();
  }

  template<typename T>
  inline agx::UInt LinkedSegmentRange<T>::size() const
  {
    return m_size;
  }

  template<typename T>
  inline agx::Bool LinkedSegmentRange<T>::empty() const
  {
    return m_size == 0ul;
  }

  template<typename T>
  inline void LinkedSegmentRange<T>::updateSizeAndState()
  {
    // Undefined to have container.end() as begin of this range.
    agxAssert( m_begin.m_segments->empty() || m_begin.m_index != m_begin.m_segments->size() );

    m_size = std::distance( m_begin, m_end );

    // If this range passes begin/end the state of the end iterator should
    // be the same as begin, enabling iterate from end to begin.
    m_begin.setBoundarylessGivenEndOfRange( m_end );
    m_end.m_boundaryless = m_begin.m_boundaryless;
  }
}
