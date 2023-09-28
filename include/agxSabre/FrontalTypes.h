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

#ifndef AGXSABRE_FRONTAL_TYPES_H
#define AGXSABRE_FRONTAL_TYPES_H

#include <queue>
#include <stack>
#include <vector>
#include <cstring>

#include <agxSabre/agxSabre.h>
#include <agxSabre/Marks.h>
#include <agxSabre/ScratchPad.h>
#include <agxSabre/SkeletonMatrix.h>
#include <agxSabre/SparseTypes.h>

#include <agx/agx.h>
#include <agx/debug.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxSabre {

  typedef UInt32PairVector  WorkVector;

  /**
  Update matrices are held by nodes when the parent is owned by another thread.
  */
  template<typename T>
  class UpdateMatrix
  {
    public:
      UpdateMatrix();

      void clear();

      size_t getSize();

      /**
      Configure the UpdateMatrix to
      */
      void configure(size_t index, const SparseMatrix<T>& matrix, T* data);

      T* operator[](size_t);
      T* find(unsigned int i, unsigned int j, unsigned int& startFrom );

      /**
      Datatype defining the different blocks held by the update matrix.
      16 bytes on 64bit arch.
      */
      struct BlockTriplet {
        T* m_ptr;
        unsigned int m_row;
        unsigned int m_col;
      };


      /**
      */
      const BlockTriplet& getBlock( unsigned int index ) { return m_blocks[index]; }

    private:
      // disabled
      UpdateMatrix( const UpdateMatrix<T>& ) {}
      UpdateMatrix<T>& operator=(const UpdateMatrix<T>& ) {return *this;}

      std::vector< BlockTriplet > m_blocks;
  };





  /**
  MfNode holds all the data needed for handling one column of the matrix.
  */
  template<typename T>
  class MfNode
  {
    public:
      MfNode();
      ~MfNode();

      UpdateMatrix<T>* getUpdateMatrix();

      /**
      */
      void setParent( unsigned int p );

      /**
      */
      unsigned int getParent() const;

      /**
      */
      unsigned int getCost() const;

      /**
      */
      void setCost( unsigned int c );

      /**
      */
      void addCost( unsigned int c );

      /**
      */
      unsigned int getLocalCost() const;

      /**
      */
      void setLocalCost( unsigned int c );


      /**
      */
      unsigned int getTotalCost() const;

      /**
      */
      void setOwner( unsigned int owner );

      /**
      */
      unsigned int getOwner()  const;

      /**
      */
      unsigned char getMixedOwnership() const;

      void  setMixedOwnership( char flag );

      void clear();


      /**
      Configure the node, used mainly by MultifrontManager.
      */
      void configure( UpdateMatrix<T>* update );


      /**
      */
      void setDone( unsigned char val );

      /**
      */
      unsigned char getDone() const;
    private:

      // disabled
      MfNode( const MfNode<T>& ) {}
      MfNode<T>& operator=( const MfNode<T>& ) {return *this;}

      UpdateMatrix<T> *m_update;

      unsigned int  m_parent;     // Parent node
      unsigned int  m_cost;       // Est. factorize cost
      unsigned int  m_localCost;

      unsigned int  m_owner;      // job id in work vector

      unsigned char m_done;       //
      unsigned char m_mixed;      // Mix of owners, include update matrices from children if present
  };



  /**
  The MultifrontManager holds the nodes and is responsible
  for memory allocations for fronts + alignment.
  */
  template<typename T>
  class MultifrontManager {

    public:
      MultifrontManager();

      ~MultifrontManager();

      /**
      */
      MfNode<T>& operator[]( size_t );


      /**
      Adds the UpdateMatrix for node "updateCol" to the Front of "frontCol" and upwards as long as the nodes
      have the same owner as frontCol. If/when a different owner is encountered, the remaining data is written
      to the UpdateMatrix of the last node seen with the correct owner.
      */
      void addUpdateToNode( const SparseMatrix<T>& L, size_t frontCol, size_t updateCol );


      /**
      Clears everything.
      */
      void clear();

      /**
      Clears data in fronts and update matrices.
      */
      void clearMatrixData();


      /**
      Set the size. Will reallocate fronts and nodes if needed.
      */
      void setSize( size_t s );

      /**
      Returns the current size.
      Number of allocated nodes/fronts can be larger than this number.
      */
      size_t getSize() const;


      /**
      splitWork will traverse the tree and estimate cost for factorizing each node and use those numbers to
      split the work.
      */
      void splitWork( const SparseMatrix<T>& matrix, const EliminationTree& tree, Marks& marks, unsigned int numWorkers );


      /**
      Will allocate fronts for factorizing a blockmatrix with the structure defined matrix with size according
      to dimension.
      */
      void allocateUpdateMatrices( const SparseMatrix<T>& matrix, ScratchPad& pad );

      /**
      Calculates the number of elements needed (including padding)
      for holding the update matrix associated with node n.
      */
      size_t calculateUpdateMatrixSize( size_t n, const SparseMatrix<T>& matrix );


      /**
      \return The vector describing how to work should be split
      */
      const WorkVector& getWorkVector() const;

    private:
      // disabled
      MultifrontManager( const MultifrontManager<T>& ) {}
      MultifrontManager<T>& operator=( const MultifrontManager<T>& ) {return *this;}

      T* m_updateMem;        // alloced
      T* m_updateData;       // aligned

      MfNode<T>*  m_nodes;   //

      UpdateMatrix<T>*         m_updateMatrices;

      WorkVector  m_work;

      size_t m_n;            // size
      size_t m_numAllocatedNodes;
      size_t m_numAllocatedUpdates;
      size_t m_numAllocatedUpdateNumbers;
  };



  //
  // UpdateMatrix
  //
  template<typename T>
  UpdateMatrix<T>::UpdateMatrix()
  {

  }






  template<typename T>
  AGX_FORCE_INLINE T* UpdateMatrix<T>::operator[]( size_t index )
  {
    return m_blocks[index].m_ptr;
  }

  template<typename T>
  T* UpdateMatrix<T>::find(unsigned int i, unsigned int j, unsigned int& startFrom )
  {
    T* ptr = 0;
    for ( ; startFrom < m_blocks.size(); ++startFrom )
    {
      if ( m_blocks[startFrom].m_row == i && m_blocks[startFrom].m_col == j )
      {
        ptr = m_blocks[startFrom].m_ptr;
        break;
      }
    }
    return ptr;
  }

  template<typename T>
  void UpdateMatrix<T>::clear()
  {
    m_blocks.clear();
  }

  template<typename T>
  size_t UpdateMatrix<T>::getSize()
  {
    return m_blocks.size();
  }



  template<typename T>
  void UpdateMatrix<T>::configure(size_t index, const SparseMatrix<T>& matrix, T* data)
  {
    const MetaLayout_t& colLayout  = matrix.getColumnLayout();
    const unsigned int* dimensions = matrix.getBlockRowDimensions();

    unsigned int start = colLayout.colPointers[ index ] + 1;
    unsigned int stop  = colLayout.colPointers[ index+1 ];

    m_blocks.reserve(  (1+stop-start) * (stop-start) / 2 );

    // Loop over blocks in column 'index'
    //
    for ( unsigned int i = start; i < stop; ++i )
    {
      unsigned int ic = colLayout.rowIndices[ i ];
      size_t iSize = paddedSize<T>( dimensions[ ic ] );

      for ( unsigned int j = i; j < stop; ++j )
      {
        BlockTriplet b;

        unsigned int jc = colLayout.rowIndices[j];

        b.m_col = ic;
        b.m_row = jc;
        b.m_ptr = data;

        m_blocks.push_back( b );

        data += iSize * paddedSize<T>( dimensions[ jc ] );
      }
    }
  }






  //
  // MfNode
  //
  template<typename T>
  MfNode<T>::MfNode() : m_update(0), m_parent(0), m_cost(0), m_owner(0), m_done(0), m_mixed(0)
  {
  }

  template<typename T>
  MfNode<T>::~MfNode()
  {
  }


  template<typename T>
  AGX_FORCE_INLINE UpdateMatrix<T>* MfNode<T>::getUpdateMatrix()
  {
    return m_update;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::setParent( unsigned int p )
  {
    m_parent = p;
  }


  template<typename T>
  AGX_FORCE_INLINE unsigned int MfNode<T>::getParent() const
  {
    return m_parent;
  }

  template<typename T>
  AGX_FORCE_INLINE unsigned int MfNode<T>::getCost() const
  {
    return m_cost;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::setCost( unsigned int c )
  {
    m_cost = c;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::addCost( unsigned int c )
  {
    m_cost += c;
  }

  template<typename T>
  AGX_FORCE_INLINE unsigned int MfNode<T>::getLocalCost() const
  {
    return m_localCost;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::setLocalCost( unsigned int c )
  {
    m_localCost = c;
  }


  template<typename T>
  AGX_FORCE_INLINE unsigned int MfNode<T>::getTotalCost() const
  {
    return m_cost + m_localCost;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::setOwner( unsigned int owner )
  {
    m_owner = owner;
  }

  template<typename T>
  AGX_FORCE_INLINE unsigned int MfNode<T>::getOwner() const
  {
    return m_owner;
  }


  template<typename T>
  AGX_FORCE_INLINE unsigned char MfNode<T>::getMixedOwnership() const
  {
    return m_mixed;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::setMixedOwnership( char flag )
  {
    m_mixed = flag;
  }

  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::clear()
  {
    m_update = 0;
    m_parent = 0;
    m_cost = 0;
    m_localCost = 0;
    m_owner = 0;
    m_done = 0;
    m_mixed = 0;
  }


  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::setDone( unsigned char val )
  {
    m_done = val;
  }

  template<typename T>
  AGX_FORCE_INLINE unsigned char MfNode<T>::getDone() const
  {
    return m_done;
  }


  template<typename T>
  AGX_FORCE_INLINE void MfNode<T>::configure( UpdateMatrix<T>* update )
  {
    m_update = update;
  }



  //
  //  MultifrontManager
  //
  template<typename T>
  MultifrontManager<T>::MultifrontManager() : m_updateMem(0), m_updateData(0),
                                              m_nodes(0), m_updateMatrices(0), m_n(0),
                                              m_numAllocatedNodes(0), m_numAllocatedUpdates(0),
                                              m_numAllocatedUpdateNumbers(0)
  {
  }


  template<typename T>
  MultifrontManager<T>::~MultifrontManager()
  {
    if ( m_nodes )
      delete[] m_nodes;

    if ( m_updateMatrices )
      delete[] m_updateMatrices;

    if ( m_updateMem )
      delete[] m_updateMem;
  }


  template<typename T>
  AGX_FORCE_INLINE MfNode<T>& MultifrontManager<T>::operator[](size_t i)
  {
    return m_nodes[i];
  }







  template<typename T>
  void MultifrontManager<T>::clear()
  {
    for ( size_t i = 0; i < m_n; ++i)
    {
      m_nodes[i].clear();
    }

    size_t uLimit = std::min(m_numAllocatedUpdates, m_n);
    for ( size_t i = 0; i < uLimit; ++i )
      m_updateMatrices[i].clear();

    memset( m_updateMem, 0, sizeof(T) * m_numAllocatedUpdateNumbers );
    m_n = 0;

    m_work.clear();
  }


  template<typename T>
  void MultifrontManager<T>::clearMatrixData()
  {
    memset( m_updateMem, 0, sizeof(T) * m_numAllocatedUpdateNumbers );

    for ( size_t i = 0; i < m_n; ++i)
      m_nodes[i].setDone( 0 );
  }


  template<typename T>
  void MultifrontManager<T>::setSize( size_t s )
  {
    // possible realloc of structures depending on N
    if ( s > m_numAllocatedNodes )
    {
      m_numAllocatedNodes = s;

      if ( m_nodes )
        delete[] m_nodes;
      m_nodes = new MfNode<T>[ m_numAllocatedNodes ];
    }

    m_n = s;
  }


  template<typename T>
  AGX_FORCE_INLINE size_t MultifrontManager<T>::getSize() const
  {
    return m_n;
  }





  template< typename T >
  AGX_FORCE_INLINE const WorkVector& MultifrontManager<T>::getWorkVector() const
  {
    return m_work;
  }




}

DOXYGEN_END_INTERNAL_BLOCK()

#endif

