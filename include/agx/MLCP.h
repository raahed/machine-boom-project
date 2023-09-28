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

#ifndef AGX_MLCP_H
#define AGX_MLCP_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable: 4512 ) // Disable warning about assignment operator could not be generated due to reference members.
#endif


#include <agx/agxPhysics_export.h>
#include <agx/agx_valarray_types.h>
#include <agx/SparseMatrix.h>
#include <agx/agx_vector_types.h>
#include <agxSabre/DenseTypes.h>



namespace agx
{
  typedef HashTable< UInt, Real* > UIntRealPtrTable;
  class SparseRangeReal;

  /**
  Class to allocate an array of elements where some other structure works
  on pointer. Suitable for solvers.
  */
  template< typename T >
  class RowAllocator
  {
    public:
      /// Constructor
      RowAllocator() : m_elements( nullptr ), m_size( 0 ), m_counter( 0 ), m_currentElement( 0 )  {}

      ~RowAllocator() { deallocate(); }

      /**
      Allocate a buffer of given size with initial value val.
      \param size - size of the buffer
      \param val - initial value to all elements (integer)
      */
      void allocate( size_t size, int val )
      {
        allocate( size );
        if ( m_elements )
          memset( m_elements, val, m_size * sizeof( T ) );
      }

      /**
      Allocate buffer of given size. The elements are not initialized.
      \param size - size of the buffer
      */
      void allocate( size_t size )
      {
        bool mustDeallocate = size > m_size;
        if ( mustDeallocate )
          deallocate();
        m_size = size;
        m_currentElement = 0;
        m_counter = 0;

        if ( m_size == 0 || !mustDeallocate ) return;

        agxAssert( m_elements == nullptr );

        m_elements = new T[ m_size ];
      }

      void allocate()
      {
        allocate( m_counter, 0 );
      }

      /**
      Deallocates the buffer and resets internal parameters.
      */
      void deallocate()
      {
        if ( m_elements ) {
          delete[] m_elements;
          m_elements = nullptr;
        }
        m_size = 0;
        m_currentElement = 0;
      }

      inline T* get( size_t numElements )
      {
        agxAssert( m_elements );
        T* elementsStart = ( m_elements + m_currentElement );
        m_currentElement += numElements;
        agxAssert( m_currentElement <= m_size );
        return elementsStart;
      }

      inline size_t& count() { return m_counter; }

    private:
      T* m_elements;
      size_t m_size;
      size_t m_counter;
      size_t m_currentElement;
  };

  DOXYGEN_START_INTERNAL_BLOCK()


  /**
  Internal class, do not export
  */
  class IdxSet
  {
    public:
      enum IdxType { EQUALITY = -1, FREE = 0, LOWER = 2, UPPER = 4, IGNORE_INDEX = 8 };

      IdxSet() : m_empty( true ) {}

      /**
      Resets this index set to default given bounds and the size of the system.
      \param bounds - vector with bounded variables
      \param size - size of the system (i.e., the number of equations)
      \note If bounds is empty no internal allocations are made, so check if this
            object is empty before using it.
      */
      void reset( const SparseRangeReal& bounds, size_t size );

      /**
      Updates this index set, ignored indices will become free. This method can
      be used for warm started systems.
      \param bounds - vector with bounded variables
      */
      void update( const SparseRangeReal& bounds );

      /**
      \return true if this index set is empty - false otherwise
      */
      inline bool empty() const { return m_empty; }

      /**
      Switch state of the indices given the current solution. E.g., if index i is free and z_i > upper_bound, i = upper.
      \param z - current solution
      \param bounds - vector containing bounded equations
      */
      void switchIndices( const RealValarray& z, const RealValarray& w, const SparseRangeReal& bounds );

      /**
      \return the number of bounded equation is this index set (update during a switchIndices call)
      */
      inline size_t getNumBoundedEquations() const { return m_atBounds.size(); }

      /**
      \return vector of mapping: local solve index -> global index, updated during a switchIndices call
      */
      inline const UIntVector& getAtBoundsVector() const { return m_atBounds; }

      /**
      \return vector of mapping: local solve index -> local solution index for all bounded equations
      */
      inline const UIntVector& getAtBoundsLocalVector() const { return m_atBoundsLocal; }

      inline IdxType& operator() ( size_t i )
      {
        agxAssert( i < m_ix.size() );
        return (IdxType&)m_ix[ i ];
      }

      inline IdxType operator() ( size_t i ) const
      {
        agxAssert( i < m_ix.size() );
        return (IdxType)m_ix[ i ];
      }

    private:
      IntValarray m_ix;                   /**< Valarray containing the current flag of the equation */
      bool        m_empty;                /**< This is empty if there's no bounds */
      UIntVector  m_atBounds;             /**< Contains global row indices */
      UIntVector  m_atBoundsLocal;        /**< Contains local row indices */
  };
  DOXYGEN_END_INTERNAL_BLOCK()

  class AGXPHYSICS_EXPORT MLCP
  {
    public:
      MLCP( SparseMatrix& H, const RealValarray& q, RealValarray& z, const SparseRangeReal& bounds, const IntValarray& signs = IntValarray() );

      /**
      Solves this mixed LCP.
      \param warm - if this method has been called before with this matrix and bounds, pass true - otherwise false.
      \return -1 if the solution failed (could still be a valid solution), otherwise the total number of LCP iterations.
      */
      int solve( bool warm = false );

      size_t getIterations() const { return m_numIterations; }

    private:
      /**
      Initial linear solve of the whole system.
      */
      void initialSolve();

      /**
      Solve for the bounded equations and updates the solution.
      */
      void solvePP();

      /**
      Solves a dense matrix problem given the current equations that are out of bounds.
      \param atBounds - int vector with mapping index of u -> index z
      \param atBoundsLocal - int vector with mapping index of u -> index in bounds
      \param bounds - vector containing the bounded equations
      \param u - current right hand side (solution is written here as well)
      */
      void solveSubproblem( const UIntVector& atBounds, const UIntVector& atBoundsLocal, const SparseRangeReal& bounds, RealValarray& u );

      void updateInverse( const UIntVector& atBounds, const SparseRangeReal& bounds );

      void buildSubmatrix( const UIntVector& atBounds, const UIntVector& atBoundsLocal, const SparseRangeReal& bounds, agxSabre::DenseMatrix<Real>& subH ) const;

      /**
      Given an unit solve solution, this method copies the wanted equations and maps it to the global row.
      */
      void allocateAndMapUnitSolveSolution( const SparseRangeReal& bounds, UInt row, const RealValarray& solution );

      /**
      Simply performs z *= signs. Make sure the matrix is symmetric before this method is used.
      */
      void convertSigns( RealValarray& z, const IntValarray& signs ) const;

      /**
      Given w < Hz + q < w (?), z*w = 0, this method computes w.
      */
      void computeSlack();

      /**
      Computes the residual. I.e., computes how far away from:
      b_l < z < b_u, w = 0,
      b_l = z, w >= 0,
      b_u = z, w <= 0 we are.
      */
      Real computeResidual() const;

    private:
      SparseMatrix&           m_H;                      /**< Sparse (blocked or non-blocked, symmetric or non-symmetric) matrix */
      const RealValarray&     m_q;                      /**< Negative right hand side (Ax = b, q = -b) */
      RealValarray&           m_z;                      /**< Final solution */
      const SparseRangeReal&  m_bounds;                 /**< Vector containing bounded equations (row_of_bounded_variable,(lower_bound,upper_bound)) */
      const IntValarray&      m_signs;                  /**< If H is symmetric, constraint equations needs -1 in this array. Other equations should have 1 */
      IdxSet                  m_idxSet;                 /**< Index set handling bounded equations */
      RealValarray            m_w;                      /**< Slack w < Hz + q < w, w*z = 0 */
      UIntRealPtrTable        m_boundSolution;          /**< Given equation row, gives solution from local solve */
      RowAllocator<Real>      m_boundSolutionAllocator; /**< Allocates local, sparse, vector of the bounded equations during unit solve */
      RealValarray            m_unitSolveSolution;      /**< Local buffer used during unit solves */
      RealValarray            m_z0;
      RealValarray            m_zBest;
      size_t                  m_numIterations;
  };
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
