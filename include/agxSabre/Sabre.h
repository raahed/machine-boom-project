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

#ifndef AGXSABRE_SABRE_H
#define AGXSABRE_SABRE_H

#include <agxSabre/agxSabre.h>
#include <agxSabre/SabreData.h>

namespace agxSabre
{

  /**
  Sabre is the main factorizer and solver.
  */
  class AGXSABRE_EXPORT Sabre
  {
    public:


      /**
      Factors the sparse matrix H into L D L'.
      */
      template< typename T >
      static void factor( SabreData<T>& sabredata );


      /**
      Solves the equation A x = b
      Using the previous factored matrix.
      \param sabredata Structure with data needed by sabre
      \param b Right hand side vector, at least as long as number of rows in factored matrix
      \param x Result vector
      */
      template< typename T >
      static bool solve( SabreData<T>& sabredata, const T* b, T* x );

      /**
      Solves the equation A x = b inplace.
      Using the previous factored matrix.
      \param sabredata Structure with data needed by sabre
      \param b Right hand side vector, overwritten with the result.
      */
      template< typename T >
      static bool solve( SabreData<T>& sabredata, T* b );


      /**
      Read an equation from the block matrix and store the output in C(:,cIndex)
      \param C DenseMatrix, output
      \param cIndex Which column in C to write
      \param equation Pair with block row index and row index in block
      */
      template< typename T >
      static void readEquation( agxSabre::SabreData<T>& sabredata,
                                agxSabre::DenseMatrix<T>& C,
                                uint32_t cIndex,
                                const agxSabre::UInt32Pair& equation );


      /**
      Use the data in getEquationRemovalVector and getEquationInsertionVector and modify the matrix
      factorization accordingly. This is the recommended way to perform the operation.
      */
      template< typename T >
      static bool modifyMatrix( SabreData<T>& sabredata, enum MatrixMethod = MATRIX_AUTO_SELECT_METHOD );


      /**
      It is highly recommended to use modifyMatrix() to update the matrix factorization and not
      use this method. Incorrect usage can destroy the factorization.

      If a modification routine such as partialFactor should be called, then this method
      must be used first to prepare internal structures.
      */
      template< typename T >
      static void prepareForModification( SabreData<T>& sabredata );


      /**
      \param If non-null pointer is sent in, an estimate on the number om gemm calls
             used for factorizing the matrix will be stored.
      \return Number of fill-ins for the matrix
      */
      template< typename T >
      unsigned int estimateWork( SabreData<T>& sabredata, unsigned int* gemCallsPtr );



      /**
      Solves the equation A x = b inplace.
      Using the previous factored matrix.
      \param sabredata Structure with data needed by sabre
      \param b Right hand side vector, overwritten with the result.
      \param startRow start row >= 0 if known, otherwise -1 to search for it
      \return the lowest index in solution where data has been written (ret == getNumRows if invalid)
      */
      template< typename T >
      size_t unitSolve( SabreData<T>& sabredata, T* b, int startRow = -1 );


      /**
      For solution Ax=b, do numIterations steps of iterative refinement.
      */
      void iterativeRefinement( SabreData<double>& sabredata, const double* b, double* x, int numIterations );


      /**
      Compute which method that is likely to fastest for computing a new factorization.
      */
      template< typename T >
      static MatrixMethod selectMethod( SabreData<T>& sabredata );


      /**
      For all block row indices in container, mark elements in buffer that would be
      affected given a certain EliminationTree strucutre.
      */
      static void markDirtyNodes( const class agxSabre::EliminationTree& tree,
                                  const agxSabre::UInt32PairVector& container,
                                  unsigned char* buffer );


  };





  /**
  Performs L D L' factorization of blk.
  \param block to be factored, will be overwritten with L
  \param d Pointer to memory where D will be stored
  */
  template< typename T >
  AGX_FORCE_INLINE void ldlt( DataBlock<T>& block, T* d)
  {
    agx::prefetch<agx::L1>( d );
    const size_t n = block.getNumColumns();

    for (size_t i =0; i < n; ++i )
    {
      d[i] = block(i, i);
      block(i,i) = 1;
    }

    // factor
    for (size_t i=1; i<n; ++i)
    {
      for (size_t j = 0; j<i; ++j)
      {
        for ( size_t tmp=0; tmp<j; ++tmp )
          block(i,j) -= block(i,tmp) * d[tmp] * block(j,tmp);

        block(i,j) /= d[j];
      }

      for ( size_t tmp=0; tmp<i; ++tmp )
        d[i] -= block(i, tmp) * d[tmp] * block(i, tmp );
    }

    for (size_t i=0; i<n; ++i)
      for (size_t j=i+1; j<n; ++j)
        block(i,j) = 0;

    /*
    for( int j = 0; j<n; ++j)
      if ( block(j,j) < 1E-12 && block(j,j) > -1E-12 )
        std::cout << "bad diag. value" << std::endl;
    */
  }




} // namespace

#endif
