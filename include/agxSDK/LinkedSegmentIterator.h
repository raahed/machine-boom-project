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

#include <agxSDK/LinkedSegment.h>

namespace agxSDK
{
  /**
  Optionally unbounded linked structure segment iterator object to iterate segments in a SegmentStructure.
  The template parameter is the type of the implemented LinkedSegment object.
  */
  template<typename T>
  class LinkedSegmentIterator
  {
    public:
      /**
      \return begin node iterator given segments container
      */
      static LinkedSegmentIterator begin( const agxSDK::LinkedSegmentContainer& segments );

      /**
      \return end node iterator given segments container
      */
      static LinkedSegmentIterator end( const agxSDK::LinkedSegmentContainer& segments );

      /**
      \param it - reference iterator
      \param numSegments - number of segments to step forward in the list
      \return node iterator \p numSegments after \p it
      */
      static LinkedSegmentIterator incremented( const LinkedSegmentIterator& it, agx::UInt numSegments );

    public:

      /**
      Create an iterator that doesn't point anywhere. The only method that is
      safe to call on such an iterator
      */
      LinkedSegmentIterator();

      /**
      Construct given index in list, mode and node container.
      \param index - index in list
      \param boundaryless - if true it's possible to iterate passed the bounds of the node container
      \param segments - node container
      */
      LinkedSegmentIterator( agx::UInt index, agx::Bool boundaryless, const agxSDK::LinkedSegmentContainer& segments );


      /**
      Construct given segment.
      \param segment - Segment to point to.
      */
      LinkedSegmentIterator( T* segment );

      /**
      Copy constructor.
      \param other - other iterator
      */
      LinkedSegmentIterator( const LinkedSegmentIterator& other );

      DOXYGEN_START_INTERNAL_BLOCK()
      /**

      This iterator holds a reference to the segments container.
      */
      LinkedSegmentIterator& operator= ( const LinkedSegmentIterator& other );
      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      \return true if this iterator is identical to \p other
      */
      agx::Bool operator== ( const LinkedSegmentIterator& other ) const;

      /**
      \return true if this iterator is different from \p other
      */
      agx::Bool operator!= ( const LinkedSegmentIterator& other ) const;

      /**
      Increment this iterator by one. If this iterator is boundaryless and
      index + 1 == segments.size(), the index will become 0.
      \return this instance, incremented
      */
      LinkedSegmentIterator& operator++ ();

      /**
      Increment this iterator by one. If this iterator is boundaryless and
      index + 1 == segments.size(), the index will become 0.
      \return iterator copy to previous value
      */
      LinkedSegmentIterator  operator++ ( int );

      /**
      Decrement this iterator by one. If this iterator is boundaryless and
      index == 0, the index will become segments.size() - 1.
      \return this instance, decremented
      */
      LinkedSegmentIterator& operator-- ();

      /**
      Decrement this iterator by one. If this iterator is boundaryless and
      index == 0, the index will become segments.size() - 1.
      \return iterator copy to previous value
      */
      LinkedSegmentIterator  operator-- ( int );

      /**
      \return iterator incremented \p numSegments
      */
      LinkedSegmentIterator operator+ ( agx::UInt numSegments ) const;

      /**
      \return iterator incremented \p numSegments
      */
      LinkedSegmentIterator operator- ( agx::UInt numSegments ) const;

      /**
      \return the difference/distance between this and \p other
      */
      agx::UInt operator- ( const LinkedSegmentIterator& other ) const;

      /**
      \return the node this iterator is pointing at
      */
      T* operator* () const;

      T* operator-> () const;

      /**
      \return the node this iterator is pointing at
      */
      T* get() const;

      /**
      Increments this by one (same as ++it).
      \return this instance, incremented
      */
      LinkedSegmentIterator& inc();

      /**
      Decrements this by one (same as --it).
      \return this instance, decremented
      */
      LinkedSegmentIterator& dec();

      /**
      \return iterator to next element
      */
      LinkedSegmentIterator next() const;

      /**
      \return iterator to previous element
      */
      LinkedSegmentIterator prev() const;

      /**
      \return true if the iterators points to the same node
      */
      agx::Bool equals( const LinkedSegmentIterator& other ) const;

      /**
      \return the index in the segments container this iterator is referring to
      */
      agx::UInt index() const;

      /**
      \return next index given current index
      */
      agx::UInt findNextIndex( agx::UInt index ) const;

      /**
      \return previous index given current index
      */
      agx::UInt findPrevIndex( agx::UInt index ) const;

      /**
      \return true if this iterator is boundaryless and can iterate from last to first and from first to last
      */
      agx::Bool isBoundaryless() const;

      /**
      \return True if this iterator points past the end of the segments container.
      */
      agx::Bool isEnd() const;

    private:
      /**
      Safe index modulus segments.size() (handles segments.size() == 0).
      */
      static agx::UInt modSize( agx::UInt index, const LinkedSegmentContainer& segments );

    private:
      const LinkedSegmentContainer* containerPtr() const;

      template<typename U>
      friend class LinkedSegmentRange;
      void setBoundarylessGivenEndOfRange( const LinkedSegmentIterator& end );

    private:
      agx::UInt m_index;
      agx::Bool m_boundaryless;
      const LinkedSegmentContainer* m_segments;
  };

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::begin( const LinkedSegmentContainer& segments )
  {
    return LinkedSegmentIterator( 0u, false, segments );
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::end( const LinkedSegmentContainer& segments )
  {
    return LinkedSegmentIterator( segments.size(), false, segments );
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::incremented( const LinkedSegmentIterator& it,
                                                                         agx::UInt numSegments )
  {
    return LinkedSegmentIterator( it.m_index + numSegments, true, *it.m_segments );
  }



  template<typename T>
  inline LinkedSegmentIterator<T>::LinkedSegmentIterator()
      : m_index(agx::InvalidIndex)
      , m_boundaryless(false)
      , m_segments(nullptr)
  {
  }



  template<typename T>
  inline LinkedSegmentIterator<T>::LinkedSegmentIterator( agx::UInt index,
                                                          agx::Bool boundaryless,
                                                          const LinkedSegmentContainer& segments )
    : m_index( boundaryless ?
                 modSize( index, segments ) :
                 std::min<agx::UInt>( index, segments.size() ) ),
      m_boundaryless( boundaryless ),
      m_segments( &segments )
  {
  }

  template<typename T>
  LinkedSegmentIterator<T>::LinkedSegmentIterator( T* segment )
      : m_index( segment->getIndex() )
      , m_boundaryless( false )
      , m_segments( &segment->getLinkedStructure()->getSegmentsContainer() )
  {
  }

  template<typename T>
  inline LinkedSegmentIterator<T>::LinkedSegmentIterator( const LinkedSegmentIterator& other )
    : m_index( other.m_index ), m_boundaryless( other.m_boundaryless ), m_segments( other.m_segments )
  {
  }



  template<typename T>
  inline LinkedSegmentIterator<T>& LinkedSegmentIterator<T>::operator=(const LinkedSegmentIterator<T>& other)
  {
    this->m_index = other.m_index;
    this->m_boundaryless = other.m_boundaryless;
    this->m_segments = other.m_segments;

    return *this;
  }



  template<typename T>
  inline agx::Bool LinkedSegmentIterator<T>::operator== ( const LinkedSegmentIterator& other ) const
  {
    return m_index == other.m_index && containerPtr() == other.containerPtr();
  }

  template<typename T>
  inline agx::Bool LinkedSegmentIterator<T>::operator!= ( const LinkedSegmentIterator& other ) const
  {
    return !( *this == other );
  }

  template<typename T>
  inline LinkedSegmentIterator<T>& LinkedSegmentIterator<T>::operator++ ()
  {
    m_index = findNextIndex( m_index );
    return *this;
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::operator++ ( int )
  {
    LinkedSegmentIterator it( *this );
    ++(*this);
    return it;
  }

  template<typename T>
  inline LinkedSegmentIterator<T>& LinkedSegmentIterator<T>::operator-- ()
  {
    m_index = findPrevIndex( m_index );
    return *this;
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::operator-- ( int )
  {
    LinkedSegmentIterator it( *this );
    --(*this);
    return it;
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::operator+ ( agx::UInt numSegments ) const
  {
    const auto newIndex = m_index + numSegments;
    if ( m_boundaryless )
      return LinkedSegmentIterator( modSize( newIndex, *m_segments ), true, *m_segments );
    return LinkedSegmentIterator( std::min<agx::UInt>( newIndex, m_segments->size() ), false, *m_segments );
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::operator- ( agx::UInt numSegments ) const
  {
    if ( numSegments > m_index )
      return m_boundaryless ?
               LinkedSegmentIterator( m_segments->size() + m_index - numSegments, true, *m_segments ) :
               LinkedSegmentIterator::begin( *m_segments );
    return LinkedSegmentIterator( m_index - numSegments, m_boundaryless, *m_segments );
  }

  template<typename T>
  inline agx::UInt LinkedSegmentIterator<T>::operator- ( const LinkedSegmentIterator& other ) const
  {
    agxAssert( containerPtr() == other.containerPtr() );
    return other.m_index <= m_index ?
             m_index - other.m_index :
             m_index + ( m_segments->size() - other.m_index );
  }

  template<typename T>
  inline T* LinkedSegmentIterator<T>::operator* () const
  {
    return get();
  }

  template<typename T>
  inline T* LinkedSegmentIterator<T>::operator-> () const
  {
    return get();
  }

  template<typename T>
  inline T* LinkedSegmentIterator<T>::get() const
  {
    return (*m_segments)[ m_index ]->template as<T>();
  }

  template<typename T>
  inline LinkedSegmentIterator<T>& LinkedSegmentIterator<T>::inc()
  {
    return ++(*this);
  }

  template<typename T>
  inline LinkedSegmentIterator<T>& LinkedSegmentIterator<T>::dec()
  {
    return --(*this);
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::next() const
  {
    LinkedSegmentIterator copy( *this );
    return ++copy;
  }

  template<typename T>
  inline LinkedSegmentIterator<T> LinkedSegmentIterator<T>::prev() const
  {
    LinkedSegmentIterator copy( *this );
    return --copy;
  }

  template<typename T>
  inline agx::Bool LinkedSegmentIterator<T>::equals( const LinkedSegmentIterator& other ) const
  {
    return (*this) == other;
  }

  template<typename T>
  inline agx::UInt LinkedSegmentIterator<T>::index() const
  {
    return m_index;
  }

  template<typename T>
  inline agx::UInt LinkedSegmentIterator<T>::findNextIndex( agx::UInt index ) const
  {
    agxAssert( m_boundaryless || index < m_segments->size() );
    return m_boundaryless ?
             modSize( index + 1, *m_segments ) :
             index + 1;
  }

  template<typename T>
  inline agx::UInt LinkedSegmentIterator<T>::findPrevIndex( agx::UInt index ) const
  {
    agxAssert( m_boundaryless || index > 0 );
    return m_boundaryless &&
           index == 0 ?
             m_segments->size() - 1 :
             index - 1;
  }

  template<typename T>
  inline agx::Bool LinkedSegmentIterator<T>::isBoundaryless() const
  {
    return m_boundaryless;
  }


  template <typename T>
  inline agx::Bool LinkedSegmentIterator<T>::isEnd() const
  {
    return m_index == agx::InvalidIndex || m_segments == nullptr || m_index == m_segments->size();
  }

  template<typename T>
  inline agx::UInt LinkedSegmentIterator<T>::modSize( agx::UInt index, const LinkedSegmentContainer& segments )
  {
    return segments.empty() ?
             0ul :
             ( index % segments.size() );
  }

  template<typename T>
  inline const LinkedSegmentContainer* LinkedSegmentIterator<T>::containerPtr() const
  {
    return m_segments;
  }

  template<typename T>
  inline void LinkedSegmentIterator<T>::setBoundarylessGivenEndOfRange( const LinkedSegmentIterator& end )
  {
    m_boundaryless = m_index > end.m_index;
  }
}

DOXYGEN_START_INTERNAL_BLOCK()
namespace std
{
  template<typename T>
  struct iterator_traits<agxSDK::LinkedSegmentIterator<T>>
  {
    using difference_type = agx::UInt;
    using value_type = T*;
    using iterator_category = std::random_access_iterator_tag;
  };
}
DOXYGEN_END_INTERNAL_BLOCK()

