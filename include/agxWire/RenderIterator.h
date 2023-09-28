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

#include <agxWire/Node.h>

namespace agxWire
{
  /**
  Proxy class for wire iterators.
  */
  class AGXPHYSICS_EXPORT RenderIterator
  {
    public:
      RenderIterator( agxWire::NodeConstIterator it );
      RenderIterator( const agxWire::RenderIterator& other );
      ~RenderIterator();

      agxWire::RenderIterator&  operator=  ( const agxWire::RenderIterator& other );
      const agxWire::NodeRef&   operator*  () const;
      const agxWire::NodeRef*   operator-> () const;

      bool                      operator== ( const agxWire::RenderIterator& other ) const;
      bool                      operator!= ( const agxWire::RenderIterator& other ) const;
      agxWire::RenderIterator&  operator++ ();
      agxWire::RenderIterator&  operator-- ();
      agxWire::RenderIterator   operator++ (int);
      agxWire::RenderIterator   operator-- (int);

      /// same as ==, for SWIG
      bool                      equal ( const agxWire::RenderIterator& other ) const;

      /**
      Increment this by one.
      \return incremented iterator
      */
      agxWire::RenderIterator& inc();

      /**
      Decrement this by one.
      \return decremented iterator
      */
      agxWire::RenderIterator& dec();

      /**
      \return iterator to element after this
      */
      agxWire::RenderIterator next() const;

      /**
      \return iterator to element before this
      */
      agxWire::RenderIterator prev() const;

      /**
      \return the actual wire node container iterator
      */
      agxWire::NodeConstIterator getIterator() const;

      /**
      \return the wire node this iterator is pointing to
      */
      agxWire::Node* get() const;

    private:
      RenderIterator() {} // Not defined.

    private:
      NodeConstIterator m_it;
  };

  AGX_FORCE_INLINE RenderIterator::RenderIterator( agxWire::NodeConstIterator it )
    : m_it( it )
  {
  }

  AGX_FORCE_INLINE RenderIterator::RenderIterator( const RenderIterator& other )
    : m_it( other.m_it )
  {
  }

  AGX_FORCE_INLINE RenderIterator::~RenderIterator()
  {
  }

  AGX_FORCE_INLINE RenderIterator& RenderIterator::operator=( const RenderIterator& other )
  {
    m_it = other.m_it;
    return *this;
  }

  AGX_FORCE_INLINE const NodeRef& RenderIterator::operator*() const
  {
    return *m_it;
  }

  AGX_FORCE_INLINE const NodeRef* RenderIterator::operator->() const
  {
    return &(*m_it);
  }

  AGX_FORCE_INLINE bool RenderIterator::equal( const RenderIterator& other ) const
  {
    return m_it == other.m_it;
  }

  AGX_FORCE_INLINE bool RenderIterator::operator==( const RenderIterator& other ) const
  {
    return m_it == other.m_it;
  }

  AGX_FORCE_INLINE bool RenderIterator::operator!=( const RenderIterator& other ) const
  {
    return !(*this == other);
  }

  inline RenderIterator& RenderIterator::operator++()
  {
    return inc();
  }

  inline RenderIterator& RenderIterator::operator--()
  {
    return dec();
  }

  inline RenderIterator RenderIterator::operator++(int)
  {
    return RenderIterator( m_it++ );
  }

  inline RenderIterator RenderIterator::operator--(int)
  {
    return RenderIterator( m_it-- );
  }

  AGX_FORCE_INLINE RenderIterator& RenderIterator::inc()
  {
    ++m_it;
    return *this;
  }

  AGX_FORCE_INLINE RenderIterator& RenderIterator::dec()
  {
    --m_it;
    return *this;
  }

  inline RenderIterator RenderIterator::next() const
  {
    NodeConstIterator it = m_it;
    return RenderIterator( ++it );
  }

  inline RenderIterator RenderIterator::prev() const
  {
    NodeConstIterator it = m_it;
    return RenderIterator( --it );
  }

  inline NodeConstIterator RenderIterator::getIterator() const
  {
    return m_it;
  }

  inline agxWire::Node* RenderIterator::get() const
  {
    return *m_it;
  }
}

