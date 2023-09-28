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

#ifndef AGXSABRE_SABREDATA_H
#define AGXSABRE_SABREDATA_H

#include <agxSabre/agxSabre.h>

#include <agx/Prefetch.h>
#include <agx/HashTable.h>
#include <agx/HashFunction.h>
#include <agx/Timer.h>
#include <agx/Thread.h>
#include <agx/LinearProbingHashTable.h>
#include <agx/QuadraticProbingHashTable.h>

#include <agxSabre/export.h>
#include <agxSabre/SparseTypes.h>
#include <agxSabre/DenseTypes.h>
#include <agxSabre/EliminationTree.h>
#include <agxSabre/FrontalTypes.h>
#include <agxSabre/Permutation.h>
#include <agxSabre/SkeletonMatrix.h>
#include <agxSabre/Marks.h>

#include <agxSabre/Reordering.h>
#include <agxSabre/ScratchPad.h>


#include <HashImplementationSwitcher.h>


namespace agxSabre
{

  // maps logical i,j to blockindex + transpose
  #if HASH_FOR_SABRE == HASH_NEW
  using BlockLookupType = agx::LinearProbingHashTable<std::pair<int,int>, int>;
  #elif HASH_FOR_SABRE == HASH_OLD
  typedef agx::QuadraticProbingHashTable< std::pair<int,int>, int > BlockLookupType;
  #else
    #error
  #endif



  /**
  A collection of datastructes needed for different linear algebra
  operations provided by Sabre.

    Example usage:
    - Call clear() or use a default constructed object.
    - Describe the matrix layout with setLogicalStructure( ... )
    - Fill matrix with data:
       - loop: getBlockPointer( i, j, transpose, leadingDimension )
    - Use Sabre, e.g. Sabre::factor( sabredata ) and Sabre::solve( sabredata, x )

  */
  template< typename T >
  class SabreData
  {
    public:

      /**
      Default constructor for SabreData. This is the one that is recommended to use.
      */
      SabreData();

      /**
      */
      SabreData( size_t blockSize, size_t i, size_t j, size_t nnz, int* r, int* c, double* data );
      SabreData( size_t* blkSizes, size_t numBlockRows, size_t i, size_t j, size_t nnz, int* r, int* c, double* data );

      SabreData( Factorizer factorizer, unsigned int numThreads, size_t* blkSizes, size_t numBlockRows, size_t i, size_t j, size_t nnz, int* r, int* c, double* data, uint32_t numBodies = 0 );



      SparseMatrix<T>&       getMatrix();
      const SparseMatrix<T>& getMatrix() const;

      SparseMatrix<T>&       getLMatrix();
      const SparseMatrix<T>& getLMatrix() const;

      DenseVector<T>&        getDVector();
      const DenseVector<T>&  getDVector() const;

      EliminationTree&       getEliminationTree();
      const EliminationTree& getEliminationTree() const;

      MultifrontManager<T>&       getMultifrontManager();
      const MultifrontManager<T>& getMultifrontManager() const;

      /**
      \return The unpermuted skeleton matrix describing the layout of the matrix that's being factored
      */
      SkeletonMatrix&       getSkeleton();
      const SkeletonMatrix& getSkeleton() const;

      SkeletonMatrix&       getSkeletonPermuted();
      const SkeletonMatrix& getSkeletonPermuted() const;


      /**
      \return The marks object used for this set of sabredata.
      */
      Marks&       getMarks();
      const Marks& getMarks() const;


      ScratchPad& getScratchPad();


      /**
      Structure with equation number which should be removed from the factorization.
      \see Sabre::modifyMatrix
      */
      agxSabre::UInt32Vector& getEquationRemovalVector();

      /**
      Structure with equation numbers which should be put back in the factorization.
      \see Sabre::modifyMatrix
      */
      agxSabre::UInt32Vector& getEquationInsertionVector();


      /**
      Clear both the deletion and insertion vector
      \see getEquationRemovalVector
      \see getEquationInsertionVector
      */
      void clearEquationVectors();


      /**
      Check if equation i is removed from the factorization
      */
      bool isEquationRemoved( unsigned int i ) const;


      /**
      */
      bool haveRemovedEquations() const;

      /**
      Clears all internal structures.
      */
      void clear();


      /**
      return true if the matrix is factored.
      */
      inline bool isFactored() const;


      /**
      Sets the logical structure of the matrix to be factorized.
      The method takes a lower triangular matrix in packed row format.
      \param n Number of block rows
      \param pointers Row pointers
      \param indices Column indices
      \param dimensions Number of rows for each block row
      \param numThreads hint about how many threads that can be used. Value is used to determine which factorizer to use.
      */
      bool setLogicalStructure( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions, unsigned int numThreads=1 );

      /**
      Returns a pointed where block (i,j) should be written.
      \param i Block row index
      \param j Block column index
      \param writeTransposed Return value indicating if the data should be written in transposed order
      */
      T* getBlockPointer( size_t i, size_t j, bool& writeTransposed, unsigned int& leadingDimension );


      /**
      \return The permutation data being used
      */
      Permutation& getPermutation();

      /**
      \return The permutation data being used
      */
      const Permutation& getPermutation() const;


      /**
      \return The amount of memory used for the internal structures
      */
      unsigned int getMemoryUsage() const;

      /**
      Sets the number of bodies (i.e safe pivots).
      Must be called before setLogicalStructure.
      */
      void setNumBodies( unsigned int num );

      /**
      \return The number of bodies (i.e. safe pivots)
      */
      unsigned int getNumBodies() const;


      /**
      \return Number of threads to use.
      */
      unsigned int getNumThreads() const;

      /**
      Set the maximum number of threads the factorizer can use.
      Value will be clamped by agx::getNumThreads

      Only used by the Multifrontal factorizer.
      \see setFactorizer
      */
      void setNumThreads( unsigned int numThreads );


      /**
      \return Which factorizer that should be used on the matrix.
      */
      Factorizer getFactorizer() const;


      /**
      Sets which factorize method that should be used.
      */
      void setFactorizer( Factorizer f );


      /// \return The factor time, for profiling
      agx::Real getFactorTime() const;

      /// Store amount of time needed to perform factor
      void setFactorTime( agx::Real t );

      /// \return The solve time, for profiling
      agx::Real getSolveTime() const;

      /// Store amount of time needed for solve operation
      void setSolveTime( agx::Real t );



      bool haveOriginalTree() const;
      bool haveMatchingTree() const;

    private:

      friend class Sabre;
      friend class SabreFactor; // SabreFactor and SabreUpdate will be removed as friends...
      friend class SabreUpdate; // Perhaps also Sabre.



      void allocateStructures();


      void updateModificationStructures();


      Marks                m_marks;
      Permutation          m_p;
      SkeletonMatrix       m_skeleton;
      SkeletonMatrix       m_skeletonPermuted;
      EliminationTree      m_eTree;

      SparseMatrix<T>      m_A;
      SparseMatrix<T>      m_L;

      DenseVector<T>       m_D;

      BlockLookupType      m_mapper;
      MultifrontManager<T> m_manager;
      ScratchPad           m_scratchPad;


      // vectors with global equation numbers
      UInt32Vector            m_toRemove;
      UInt32Vector            m_toInsert;

      // vectors with block pair numbers
      UInt32PairVector        m_removeEqs;
      UInt32PairVector        m_insertEqs;

      agx::HashSet<uint32_t>  m_removedEqSet;

      // clear helpers, used by e.g. partial factor
      UInt32Vector            m_blockRowCounts;
      UInt32Vector            m_innerIndices;


      agx::Real               m_factorTime;
      agx::Real               m_solveTime;

      unsigned int            m_numBodies;
      unsigned int            m_factorizer;
      unsigned int            m_numThreads;

      bool                    m_factored;

      bool                    m_originalTree;
      bool                    m_matchingTree;

  };




  template< typename T >
  SabreData<T>::SabreData() :
    m_factorTime(0), m_solveTime(0),  m_numBodies(0), m_factorizer( SERIAL ),
    m_numThreads(1), m_factored( false ),
    m_originalTree( true ), m_matchingTree( true )
  {
  }


  template< typename T >
  SabreData<T>::SabreData( size_t blockSize, size_t i, size_t j, size_t nnz, int* r, int* c, double* data ) :
    m_A(blockSize, i, j, nnz, r, c, data ), m_factorTime(0), m_solveTime(0),
    m_numBodies(0), m_factorizer( SERIAL ), m_numThreads(1), m_factored( false ),
    m_originalTree( true ), m_matchingTree( true )
  {
    // setLogicalStructure computes the elimination tree, that method is not being called when using
    // sabre this way.
    m_eTree.clear();
    m_eTree.computeFillIns( m_A );

    m_blockRowCounts.resize( m_A.getNumBlockRows() );
    m_innerIndices.resize( m_A.getNumRows() );
  }


  template< typename T >
  SabreData<T>::SabreData( size_t* blkSizes, size_t numBlockRows, size_t i, size_t j, size_t nnz, int* r, int* c, double* data ) :
    m_A(blkSizes, numBlockRows, i, j, nnz, r, c, data ), m_factorTime(0), m_solveTime(0),
    m_numBodies(0), m_factorizer( SERIAL ), m_numThreads(1), m_factored( false ),
    m_originalTree( true ), m_matchingTree( true )
  {
    // setLogicalStructure computes the elimination tree, that method is not being called when using
    // sabre this way.
    m_eTree.clear();
    m_eTree.computeFillIns( m_A );

    m_blockRowCounts.resize( m_A.getNumBlockRows() );
    m_innerIndices.resize( m_A.getNumRows() );
  }



  template< typename T >
  SabreData<T>::SabreData( Factorizer factorizer, unsigned int numThreads, size_t* blkSizes, size_t numBlockRows,
                           size_t i, size_t j, size_t nnz, int* r, int* c, double* data, uint32_t numBodies ) :
    m_factorTime(0), m_solveTime(0), m_numBodies(numBodies), m_factorizer( factorizer ),
    m_numThreads(numThreads), m_factored( false ),
    m_originalTree( true ), m_matchingTree( true )
  {
    // temp matrix in row major format
    SparseMatrix<T> tempMatrix( blkSizes, numBlockRows, i, j, nnz, r, c, data );

    std::vector<unsigned int> dim;
    for ( size_t loop = 0; loop < numBlockRows; ++loop )
      dim.push_back( blkSizes[loop] );

    // NumBodies is default 0 which leaves the matrix unpermuted.
    //
    // We do allow for computing a permutation which adds some additional requirements on
    // the matrix structure if a safe permutation should be possible to find.
    setLogicalStructure( numBlockRows, tempMatrix.getPointerVector(), tempMatrix.getIndexVector(), &dim[0], m_numThreads );

    unsigned int ld;
    bool transposed;

    // Traverse tempMatrix and write data into this structure (which can be allocated in different order depending on m_factorizer)
    for ( size_t row = 0; row < numBlockRows; ++row )
    {
      T* ptr;
      for ( int p = tempMatrix.getRowPointer( row ); p < tempMatrix.getRowPointer( row + 1 ); ++p )
      {
        unsigned int col = tempMatrix.getColumnIndex( p );
        ptr = getBlockPointer( row, col, transposed, ld);

        memcpy( ptr, tempMatrix.getBlockPointer( p ), sizeof(T) * tempMatrix.getBlockRowPaddedDimension(row)*tempMatrix.getBlockRowPaddedDimension(col));
      }
    }
  }


  template< typename T >
  inline void SabreData<T>::setNumBodies( unsigned int num )
  {
    m_numBodies = num;
  }

  template< typename T >
  inline unsigned int SabreData<T>::getNumBodies() const
  {
    return m_numBodies;
  }


  template< typename T >
  inline unsigned int SabreData<T>::getNumThreads() const
  {
    return m_numThreads;
  }

  template< typename T >
  inline void SabreData<T>::setNumThreads( unsigned int numThreads )
  {
    m_numThreads = std::min( numThreads, (unsigned int) agx::getNumThreads() );
  }



  template< typename T >
  void SabreData<T>::clear()
  {
    m_factored = false;
    m_factorTime = m_solveTime = 0;

    m_numBodies = 0;

    m_A.clear();
    m_L.clear();

    m_D.clear();

    m_p.clear();

    m_skeleton.clear();
    m_skeletonPermuted.clear();

    m_manager.clear();
    m_mapper.clear();

    m_originalTree = true;
    m_matchingTree = true;

    m_removedEqSet.clear();
    std::fill( m_blockRowCounts.begin(), m_blockRowCounts.end(), 0 );
  }

  template< typename T >
  inline bool SabreData<T>::isFactored() const
  {
    return m_factored;
  }

  template< typename T >
  inline SparseMatrix<T>& SabreData<T>::getMatrix()
  {
    return m_A;
  }


  template< typename T >
  inline const SparseMatrix<T>& SabreData<T>::getMatrix() const
  {
    return m_A;
  }


  template< typename T >
  inline SparseMatrix<T>& SabreData<T>::getLMatrix()
  {
    return m_L;
  }

  template< typename T >
  inline const SparseMatrix<T>& SabreData<T>::getLMatrix() const
  {
    return m_L;
  }



  template< typename T >
  inline DenseVector<T>& SabreData<T>::getDVector()
  {
    return m_D;
  }

  template< typename T >
  inline const DenseVector<T>& SabreData<T>::getDVector() const
  {
    return m_D;
  }


  template< typename T >
  inline MultifrontManager<T>& SabreData<T>::getMultifrontManager()
  {
    return m_manager;
  }

  template< typename T >
  inline const MultifrontManager<T>& SabreData<T>::getMultifrontManager() const
  {
    return m_manager;
  }


  template< typename T >
  inline EliminationTree& SabreData<T>::getEliminationTree()
  {
    return m_eTree;
  }

  template< typename T >
  inline const EliminationTree& SabreData<T>::getEliminationTree() const
  {
    return m_eTree;
  }


  template< typename T >
  bool SabreData<T>::setLogicalStructure( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions, unsigned int numThreads )
  {
    bool result = true;

    if (n == 0)
      return result;

    m_numThreads = numThreads;

    m_p.setSize( n );
    m_marks.setSize( (unsigned int)n );
    m_skeleton.set( n, pointers, indices, dimensions );


    // The multifrontal impl has some overhead compared to the serial version so
    // we require the matrix to have a certain size before using the threaded
    // implementation.
    //
    if ( m_numThreads > 1 && n > 128)
      m_factorizer = MULTIFRONTAL;
    else
      m_factorizer = SERIAL;


    // We must know number of bodies to be able to use zero-swap.
    if (  m_numBodies > 0 )
    {
      result = Reordering::computeSymamdZeroSwap( m_skeleton, m_skeletonPermuted, m_eTree, m_numBodies, m_p, m_scratchPad );
    }
    else
    {
      m_p.clear();
    }

    m_p.computeInversePermutation();
    m_skeletonPermuted.permute( m_skeleton, m_p, m_scratchPad );

    /*
    if ( m_factorizer == MULTIFRONTAL )
    {
      // use postorder tree order.
      m_eTree.clear();
      m_eTree.computeTree( m_skeletonPermuted );
      m_eTree.computePostOrder();
      m_p.postOrder( m_eTree.getPostOrdering() );

      m_skeletonPermuted.permute( m_skeleton, m_p, m_scratchPad );
    }
    */

    // Elimination tree for fillins in m_L
    m_eTree.clear();
    m_eTree.computeFillIns( m_skeletonPermuted );

    // H is allocated here.
    AGX_BEGIN_TIMELINE_REPORT(allocateSystemMatrix);
    allocateStructures();
    AGX_END_TIMELINE_REPORT(allocateSystemMatrix, "Allocate System Matrix");


    m_D.setSize( m_A.getNumRows() );


    // Prepare structures for matrix modifications

    m_toRemove.reserve( n * 2 );
    m_toInsert.reserve( n * 2 );

    m_removedEqSet.reserve( m_A.getNumRows() );

    m_blockRowCounts.resize( m_A.getNumBlockRows() );
    m_innerIndices.resize( m_A.getNumRows() );


    return result;
  }


  template< typename T >
  T* SabreData<T>::getBlockPointer( size_t i, size_t j, bool& writeTransposed, unsigned int& leadingDimension )
  {
    T* res = nullptr;

    BlockLookupType::iterator it = m_mapper.find( std::make_pair( (int)i, (int)j ) );
    int p;

    if ( it != m_mapper.end() )
    {
      p = it->second;

      writeTransposed = m_skeletonPermuted.getTranspose(p);

      #if defined( AGXSABRE_DEBUG_CHECKS )
      // We must be 16-byte aligned.
      agxAssert( ((size_t) m_A.getBlockPointer(p) ) % 16 == 0 );
      #endif

      // Make sure that the leading dimension is handled correctly.
      int colId = m_A.getColumnIndex( p );

      leadingDimension = m_A.getBlockRowPaddedDimension(colId);

      #if defined( AGXSABRE_DEBUG_CHECKS )
      agxAssert( leadingDimension >= m_a.getBlockRowDimension( m_A.getColumnIndex( p ) ) );
      #endif

      res = m_A.getBlockPointer(p);
    }
    return res;
  }


  template< typename T >
  inline Permutation& SabreData<T>::getPermutation()
  {
    return m_p;
  }


  template< typename T >
  inline const Permutation& SabreData<T>::getPermutation() const
  {
    return m_p;
  }


  template< typename T >
  inline SkeletonMatrix& SabreData<T>::getSkeleton()
  {
    return m_skeleton;
  }


  template< typename T >
  inline const SkeletonMatrix& SabreData<T>::getSkeleton() const
  {
    return m_skeleton;
  }

  template< typename T >
  inline SkeletonMatrix& SabreData<T>::getSkeletonPermuted()
  {
    return m_skeletonPermuted;
  }


  template< typename T >
  inline const SkeletonMatrix& SabreData<T>::getSkeletonPermuted() const
  {
    return m_skeletonPermuted;
  }


  template< typename T >
  inline Marks& SabreData<T>::getMarks()
  {
    return m_marks;
  }


  template< typename T >
  inline const Marks& SabreData<T>::getMarks() const
  {
    return m_marks;
  }


  template< typename T >
  inline unsigned int SabreData<T>::getMemoryUsage() const
  {
    return m_A.getMemoryUsage() + m_L.getMemoryUsage();
  }


  template< typename T >
  inline ScratchPad& SabreData<T>::getScratchPad()
  {
    return m_scratchPad;
  }


  template< typename T >
  void SabreData<T>::allocateStructures()
  {
    // This is the matrix the user writes.
    size_t memory = 0;

    const size_t N = m_skeleton.getSize();

    for (size_t i = 0; i<N; ++i)
    {
      size_t rowSize = paddedSize<T>( m_skeleton.getDimension(i) );
      size_t colSize = 0;

      for (unsigned int p = m_skeleton.getRowPointer(i); p < m_skeleton.getRowPointer(i+1); ++p )
      {
        int col = m_skeleton.getColumnIndex( p );
        colSize += paddedSize<T>( m_skeleton.getDimension(col) );
      }

      memory += rowSize * colSize;
    }

    m_A.setSize( N, N, m_skeleton.getNumValues(), memory );

    m_mapper.reserve( m_skeleton.getNumValues() );


    // Traverse the permuted skeleton and allocate data in the big matrix in
    // the correct order.
    // Store unpermuted indices and pointer

    // traverse skeleton matrix
    for (int i = 0; i<(int)N; ++i)
    {
      int col;
      for (unsigned int p = m_skeletonPermuted.getRowPointer(i); p < m_skeletonPermuted.getRowPointer(i+1); ++p )
      {
        col = m_skeletonPermuted.getColumnIndex( p );
        m_A.acquireBlock( i, col, m_skeletonPermuted.getDimension(i), m_skeletonPermuted.getDimension(col), 0 );

        int I = m_p.permute(i);
        int J = m_p.permute(col);

        if ( J > I )
        {
          std::swap(I,J);
        }
        m_mapper.insert( std::make_pair( I, J ), p );
      }
    }

  }



  template< typename T >
  inline Factorizer SabreData<T>::getFactorizer() const
  {
    return (Factorizer) m_factorizer;
  }

  template< typename T >
  inline void SabreData<T>::setFactorizer( Factorizer f )
  {
    m_factorizer = f;
  }


  template< typename T>
  inline agx::Real SabreData<T>::getFactorTime() const
  {
    return m_factorTime;
  }

  template< typename T>
  inline void SabreData<T>::setFactorTime( agx::Real t )
  {
    m_factorTime = t;
  }


  template< typename T>
  inline agx::Real SabreData<T>::getSolveTime() const
  {
    return m_solveTime;
  }

  template< typename T >
  inline void SabreData<T>::setSolveTime( agx::Real t )
  {
    m_solveTime = t;
  }


  template< typename T>
  inline bool SabreData<T>::haveOriginalTree() const
  {
    return m_originalTree;
  }

  template< typename T >
  inline bool SabreData<T>::haveMatchingTree() const
  {
    return m_matchingTree;
  }



  template< typename T >
  inline agxSabre::UInt32Vector& SabreData<T>::getEquationRemovalVector()
  {
    return m_toRemove;
  }


  template< typename T >
  inline agxSabre::UInt32Vector& SabreData<T>::getEquationInsertionVector()
  {
    return m_toInsert;
  }

  template< typename T >
  inline void SabreData<T>::clearEquationVectors()
  {
    m_toRemove.clear();
    m_toInsert.clear();
  }

  template< typename T >
  inline bool SabreData<T>::isEquationRemoved( unsigned int i ) const
  {
    return m_removedEqSet.contains( i );
  }

  template< typename T >
  inline bool SabreData<T>::haveRemovedEquations() const
  {
    return !m_removedEqSet.empty();
  }

} // namespace

#endif

