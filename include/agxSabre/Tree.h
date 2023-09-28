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

#ifndef SABRE_TREE
#define SABRE_TREE

#include <agx/agx.h>

#include <agxSabre/Marks.h>
#include <agxSabre/export.h>
#include <agxSabre/Permutation.h>

#include <vector>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agxSabre
{

  /**
  Base class for trees in Sabre.
  Used as is for spanning trees and subclassed to produce
  elimination trees.

  The tree can contain several trees and is actually more like a bush.
  */
  class AGXSABRE_EXPORT Tree
  {
    public:
      Tree();
      virtual ~Tree();

      unsigned int getSize() const;
      void setSize( unsigned int size );

      /**
      Builds the siblings and firstchild structure from the parent hierarchy.
      */
      void computeTreeStructure();

      /**
      Computes the postordering of the tree.
      */
      void computePostOrder();


      /**
      \return The parent for node i. Parent(i) == i is that i is a root.
      */
      unsigned int getParent( unsigned int i) const;


      unsigned int* getParents();
      const unsigned int* getParents() const;

      int* getPostOrdering();
      const int* getPostOrdering() const;

      int* getFirstChildren();
      const int* getFirstChildren() const;

      int* getSiblings();
      const int* getSiblings() const;


      virtual void clear();

      virtual void print() const;

      const unsigned int* getRoots() const;

      unsigned int getNumRoots() const;
      void addRoot( unsigned int r );

      void writePermutation( Permutation& p );

      Tree& operator= ( const Tree& rhs );

    protected:
      virtual void allocateMemory( unsigned int size );
      virtual void releaseMemory();

      Tree( const Tree& /*other*/ ) : m_marks(), m_parent(0), m_firstChild(0), m_siblings(0),
                                  m_postOrder(0), m_size(0), m_allocatedSize(0), m_roots()
      {
      }


      Marks m_marks;

      unsigned int* m_parent;
      int* m_firstChild;
      int* m_siblings;
      int* m_postOrder;

      unsigned int m_size;
      unsigned int m_allocatedSize;

      std::vector< unsigned int > m_roots;
  };




  /////////////////////////////////////////////////////////////////////////////
  //
  // Tree


  AGX_FORCE_INLINE unsigned int Tree::getSize() const
  {
    return m_size;
  }

  inline void Tree::setSize( unsigned int size )
  {
    if ( size > m_allocatedSize )
    {
      releaseMemory();
      allocateMemory( size );
    }
    m_size = size;
  }


  AGX_FORCE_INLINE unsigned int Tree::getParent( unsigned int i) const
  {
    return m_parent[i];
  }


  AGX_FORCE_INLINE unsigned int* Tree::getParents()
  {
    return m_parent;
  }

  AGX_FORCE_INLINE const unsigned int* Tree::getParents() const
  {
    return m_parent;
  }


  AGX_FORCE_INLINE int* Tree::getPostOrdering()
  {
    return m_postOrder;
  }

  AGX_FORCE_INLINE const int* Tree::getPostOrdering() const
  {
    return m_postOrder;
  }


  AGX_FORCE_INLINE int* Tree::getFirstChildren()
  {
    return m_firstChild;
  }


  AGX_FORCE_INLINE const int* Tree::getFirstChildren() const
  {
    return m_firstChild;
  }

  AGX_FORCE_INLINE int* Tree::getSiblings()
  {
    return m_siblings;
  }

  AGX_FORCE_INLINE const int* Tree::getSiblings() const
  {
    return m_siblings;
  }


  AGX_FORCE_INLINE const unsigned int* Tree::getRoots() const
  {
    return &m_roots[0];
  }

  inline unsigned int Tree::getNumRoots() const
  {
    return (unsigned int)m_roots.size();
  }

  inline void Tree::addRoot( unsigned int r )
  {
    m_roots.push_back( r );
  }


  inline void Tree::writePermutation( Permutation& p )
  {
    p.setSize( m_size );

    int* perm = p.getPermutation();

    for (unsigned int i = 0 ; i < m_size; ++i)
    {
      perm[ m_postOrder[i] ] = (int)i;
    }
  }


} // namespace

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#endif

