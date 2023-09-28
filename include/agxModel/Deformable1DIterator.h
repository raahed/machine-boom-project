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

#ifndef AGXMODEL_DEFORMABLE1DITERATOR_H
#define AGXMODEL_DEFORMABLE1DITERATOR_H

#include <agxModel/Deformable1DNode.h>

namespace agxModel
{
  class AGXMODEL_EXPORT Deformable1DIterator
  {
    public:
      Deformable1DIterator(agxModel::Deformable1DNodeRef node);
      Deformable1DIterator( const agxModel::Deformable1DIterator& other );
      ~Deformable1DIterator();

      agxModel::Deformable1DIterator& operator=  ( const agxModel::Deformable1DIterator& other );
      const agxModel::Deformable1DNodeRef&        operator*  () const;
      const agxModel::Deformable1DNodeRef*        operator-> () const;

      agx::Bool operator== ( const agxModel::Deformable1DIterator& other ) const;
      agx::Bool operator!= ( const agxModel::Deformable1DIterator& other ) const;

      agxModel::Deformable1DIterator& operator++ ();
      agxModel::Deformable1DIterator& operator-- ();
      agxModel::Deformable1DIterator  operator++ ( int );
      agxModel::Deformable1DIterator  operator-- ( int );

      /**
      \return true if this is the begin iterator
      */
      agx::Bool isBegin() const;

      /**
      \return true if this is the end iterator
      */
      agx::Bool isEnd() const;

      /**
      Same as == to be used with exports to languages with not complete support for operator overloading.
      */
      agx::Bool equal( const agxModel::Deformable1DIterator& other ) const;

      /**
      Increment this by one.
      \return incremented iterator
      */
      agxModel::Deformable1DIterator& inc();

      /**
      Decrement this by one.
      \return decremented iterator
      */
      agxModel::Deformable1DIterator& dec();

      /**
      \return iterator to element after this
      */
      agxModel::Deformable1DIterator next() const;

      /**
      \return iterator to element before this
      */
      agxModel::Deformable1DIterator prev() const;

      /**
      \return the node this iterator is pointing to
      */
      agxModel::Deformable1DNode* get() const;

      /**
      \return the segment this iterator is pointing to
      */
      const agxModel::Segment* getSegment() const;

    private:
      Deformable1DNodeRef m_node;
  };

  inline Deformable1DIterator& Deformable1DIterator::operator= ( const Deformable1DIterator& other )
  {
    m_node = other.m_node;
    return *this;
  }

  inline const Deformable1DNodeRef& Deformable1DIterator::operator* () const
  {
    return m_node;
  }

  inline const Deformable1DNodeRef* Deformable1DIterator::operator-> () const
  {
    return &m_node;
  }

  inline agx::Bool Deformable1DIterator::operator== ( const Deformable1DIterator& other ) const
  {
    return m_node == other.m_node;
  }

  inline agx::Bool Deformable1DIterator::operator!= ( const Deformable1DIterator& other ) const
  {
    return !( *this == other );
  }

  inline Deformable1DIterator& Deformable1DIterator::operator++ ()
  {
    return inc();
  }

  inline Deformable1DIterator& Deformable1DIterator::operator-- ()
  {
    return dec();
  }

  inline Deformable1DIterator Deformable1DIterator::operator++ ( int )
  {
    Deformable1DIterator ret = *this;
    ++(*this);
    return ret;
  }

  inline Deformable1DIterator Deformable1DIterator::operator-- ( int )
  {
    Deformable1DIterator ret = *this;
    --(*this);
    return ret;
  }

  inline agx::Bool Deformable1DIterator::isEnd() const
  {
    return m_node == nullptr;
  }

  inline agx::Bool Deformable1DIterator::equal( const Deformable1DIterator& other ) const
  {
    return *this == other;
  }

  inline Deformable1DIterator& Deformable1DIterator::inc()
  {
    if ( m_node == nullptr )
      return *this;

    m_node = m_node->getNumBranches() > 0 ? static_cast< Deformable1DNode* >(m_node->getBranch(0)) : nullptr;

    return *this;
  }

  inline Deformable1DIterator& Deformable1DIterator::dec()
  {
    if ( m_node == nullptr )
      return *this;

    m_node = static_cast< Deformable1DNode* >(m_node->getParent());

    return *this;
  }

  inline Deformable1DIterator Deformable1DIterator::next() const
  {
    return Deformable1DIterator(m_node != nullptr && m_node->getNumBranches() > 0 ? static_cast< Deformable1DNode* >(m_node->getBranch(0)) : nullptr);
  }

  inline Deformable1DIterator Deformable1DIterator::prev() const
  {
    return Deformable1DIterator(m_node != nullptr ? static_cast< Deformable1DNode* >(m_node->getParent()) : nullptr);
  }

  inline Deformable1DNode* Deformable1DIterator::get() const
  {
    return m_node;
  }

  inline const Segment* Deformable1DIterator::getSegment() const
  {
    return m_node != nullptr ? m_node->getSegment() : nullptr;
  }
}

#endif // AGXMODEL_DEFORMABLE1DITERATOR_H
