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

#ifndef SABRE_ELIMINATION_TREE
#define SABRE_ELIMINATION_TREE


#include <agx/agx.h>

#include <agxSabre/export.h>
#include <agxSabre/SparseTypes.h>
#include <agxSabre/Marks.h>
#include <agxSabre/Tree.h>
#include <agxSabre/SkeletonMatrix.h>
#include <vector>


namespace agxSabre
{


  /**
  Class for holding and computing an EliminationTree
  for a SparseMatrix.
  */
  class AGXSABRE_EXPORT EliminationTree : public Tree
  {
    public:

      EliminationTree();
      virtual ~EliminationTree();

      EliminationTree& operator= (const EliminationTree& rhs);

      /**
      Clears the data structures for the tree.
      */
      void clear();

      /**
      Compute the sparse structure from
      */
      template< typename T >
      void computeFillIns( const SparseMatrix<T>& H );

      /**
      Compute the number of fillins when SkeletonMatrix s is permuted with Permutation p.
      */
      int computeFillIns( const SkeletonMatrix& s, const Permutation &p, ScratchPad& pad );

      /**
      Compute the number of fillins for SkeletonMatrix s
      */
      int computeFillIns( const SkeletonMatrix& s );


      /**
      Computes the tree for the matrix
      */
      template< typename T >
      void computeTree( const SparseMatrix<T>& H );


      void computeTree( const SkeletonMatrix& H );


      virtual void print() const;

      const std::vector<int>& getPointerVector() const;
      const std::vector<int>& getIndexVector() const;

      /**
      This should not be needed by the user.
      */
      inline void connectSets( unsigned int x, unsigned int y );

      /**
      This should not be needed by the user.
      */
      inline unsigned int findRoot( unsigned int x );


    private:
      unsigned int *m_collapsedParent;
      unsigned int *m_rank;


      std::vector<int> m_pts;
      std::vector<int> m_idx;

      // private copy constructor
      EliminationTree( const EliminationTree& other ) : Tree(other), m_collapsedParent(0), m_rank(0), m_pts(), m_idx()
      {
      }

      /**
      Changes the number of nodes and possibly allocates more memory.
      Previously held data is lost.
      */
      inline void resize( unsigned int size );


      /**
      */
      template< typename T >
      void computeSets( const SparseMatrix<T>& H );


      inline void computeSets( const SkeletonMatrix& H );

      /**
      */
      virtual void allocateMemory( unsigned int size );

      /**
      */
      virtual void releaseMemory();

  };






  inline EliminationTree::EliminationTree() :
    m_collapsedParent(0), m_rank(0), m_pts(), m_idx()
  {
  }



  inline void EliminationTree::clear()
  {
    Tree::clear();

    for (unsigned int i=0; i<m_allocatedSize; ++i)
    {
      m_collapsedParent[i] = i;
    }

    if ( m_allocatedSize )
    {
      memset(m_rank, 0, sizeof(unsigned int) * m_allocatedSize);
    }
  }


  inline void EliminationTree::resize( unsigned int size )
  {
    if ( m_allocatedSize < size )
    {
      releaseMemory();
      allocateMemory( size );
    }

    m_size = (unsigned int)size;
  }


  template< typename T >
  void EliminationTree::computeFillIns( const SparseMatrix<T>& H )
  {
    computeTree( H );

    int nnzH = 0;

    if ( m_size > 0 )
      nnzH = H.getRowPointer( H.getNumBlockRows() );

    int nnz = 0;

    int n = (int)H.getNumBlockRows();

    m_pts.clear();
    m_idx.clear();
    m_pts.reserve(n+1);
    m_idx.reserve(nnzH);
    m_pts.push_back(0);


    // go along rows
    for (int i=0; i<n; ++i)
    {

      for (size_t p=H.getRowPointer(i); p < H.getRowPointer(i+1); ++p )
      {
        // start traversal if we haven't seen this one before
        for ( int j = H.getColumnIndex(p); m_marks[j] == false &&  (int)m_parent[j] <= i;    j = m_parent[j] ) {
          m_marks.mark(j);
        }
      }

      if ( m_marks.getNumMarks() == 0 || false == m_marks[i] )
      {
        m_marks.mark(i);
      }

      m_pts.push_back( m_pts[i] + m_marks.getNumMarks() );

      m_marks.sort();

      int p0 = (int)m_pts[i];

      for (size_t p = p0; p < (size_t)m_pts[i+1]; ++p )
      {
        m_idx.push_back( m_marks.getMarkIndices()[p - p0]);
      }

      nnz += m_marks.getNumMarks();

      m_marks.clear();

    }
    m_pts[n] = nnz;
  }




  inline int EliminationTree::computeFillIns( const SkeletonMatrix& s, const Permutation &p, ScratchPad& pad )
  {
    SkeletonMatrix sp;
    sp.permute( s, p, pad );

    return computeFillIns( sp );
  }


  // SkeletonMatrix and SparseMatrix doesn't have 100% matching interfaces.
  // second impl needed.
  inline int EliminationTree::computeFillIns( const SkeletonMatrix& s )
  {
    computeTree( s );

    size_t n = s.getSize();
    int nnz = 0;
    int nnzH = 0;

    if ( m_size > 0 )
      nnzH = (int)s.getNumValues();


    m_pts.clear();
    m_idx.clear();
    m_pts.reserve(n+1);
    m_idx.reserve(nnzH);
    m_pts.push_back(0);

    // go along rows
    for (size_t i=0; i<n; ++i)
    {
      for (unsigned int p=s.getRowPointer(i); p < s.getRowPointer(i+1); ++p )
      {
        // start traversal if we haven't seen this one before
        for ( unsigned int j = s.getColumnIndex(p); m_marks[j] == false &&  m_parent[j] <= i;    j = m_parent[j] ) {
          m_marks.mark(j);
        }
      }

      if ( m_marks.getNumMarks() == 0 || false == m_marks[i] )
      {
        m_marks.mark( (unsigned int)i);
      }

      m_pts.push_back( m_pts[i] + m_marks.getNumMarks() );

      m_marks.sort();

      int p0 = m_pts[i];

      for (int p = p0; p < m_pts[i+1]; ++p )
      {
        m_idx.push_back( m_marks.getMarkIndices()[p - p0]);
      }

      nnz += m_marks.getNumMarks();

      m_marks.clear();
    }

    m_pts[n] = nnz;

    return nnz;
  }



  template< typename T >
  void EliminationTree::computeTree( const SparseMatrix<T>& H )
  {
    // setup
    if ( m_allocatedSize < H.getNumBlockRows() )
      resize( (unsigned int)H.getNumBlockRows() );

    clear();

    m_size = (unsigned int)H.getNumBlockRows();

    computeSets( H );
    computeTreeStructure();
  }


  inline void EliminationTree::computeTree( const SkeletonMatrix& H )
  {
    if ( m_allocatedSize < H.getSize() )
      resize( (unsigned int) H.getSize() );

    clear();

    m_size = (unsigned int)H.getSize();

    computeSets( H );
    computeTreeStructure();
  }


  template< typename T >
  void EliminationTree::computeSets( const SparseMatrix<T>& H )
  {
    size_t rowCount = H.getNumBlockRows();

    for (int i = 0; i < (int)rowCount; ++i )
    {
      int start = H.getRowPointer( i );
      int end = H.getRowPointer( i+1 );

      for (int k = start; k < end && (int)H.getColumnIndex(k) <= i; ++k )
      {
        int j = H.getColumnIndex(k);

        // Have element at i, j
        connectSets( findRoot(j), findRoot(i) );
      }
    }
  }


  inline void EliminationTree::computeSets( const SkeletonMatrix& H )
  {
    size_t rowCount = H.getSize();

    for (size_t i = 0; i < rowCount; ++i )
    {
      unsigned int start = H.getRowPointer( i );
      unsigned int end = H.getRowPointer( i+1 );

      for (unsigned int k = start; k < end && H.getColumnIndex(k) <= i; ++k )
      {
        unsigned int j = H.getColumnIndex(k);

        // Have element at i, j
        connectSets( findRoot( (unsigned int)j), findRoot((unsigned int)i) );
      }
    }
  }




  inline void EliminationTree::connectSets( unsigned int x, unsigned int y )
  {
    if ( x != y )
    {
      if ( m_rank[x] > m_rank[y] )
        m_parent[y] = m_collapsedParent[y] = x;
      else
      {
        m_parent[x] = m_collapsedParent[x] = y;
        if ( m_rank[x] == m_rank[y] )
          ++m_rank[x];
      }
    }
  }

  inline unsigned int EliminationTree::findRoot( unsigned int x )
  {
    if ( m_parent[x] != x )
    {
      m_collapsedParent[ x ] = findRoot( m_collapsedParent[ x ] );
    }
    return m_collapsedParent[ x ];
  }

  AGX_FORCE_INLINE const std::vector<int>& EliminationTree::getPointerVector() const
  {
    return m_pts;
  }


  AGX_FORCE_INLINE const std::vector<int>& EliminationTree::getIndexVector() const
  {
    return m_idx;
  }


  inline void EliminationTree::print() const
  {
    std::cout << "Sets (" << m_size << "):" << std::endl;
    std::cout << "Node\tParent\tCol.Prnt\tPostOrder" << std::endl;
    for (size_t i=0; i<m_size; ++i)
    {
      std::cout << i << "\t" << m_parent[i] << "\t" << m_collapsedParent[i] << "\t" << m_postOrder[i] << std::endl;
    }

    std::cout << "Fill ins: " << m_idx.size() << std::endl;
  }


} // namespace

#endif

