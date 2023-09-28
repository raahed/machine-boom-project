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

#ifndef AGXSABRE_SABRESOLVE_H
#define AGXSABRE_SABRESOLVE_H

/**
\file SabreSolve.h

*/

#include <agxSabre/agxSabre.h>
#include <agxSabre/SabreData.h>

namespace agxSabre
{


  class AGXSABRE_EXPORT SabreSolve
  {

    public:

      /**
      Normal solve.
      */
      static bool solve( const agxSabre::SabreData<double>& sd, agxSabre::DenseVector<double>& x );

      /**
      Normal solve.
      */
      static bool solve( const agxSabre::SabreData<double>& sd, double* x );




      /**
      Solve inplace M x_out = x_in  using M(1:len,1:len)

      Reduce amount of work by starting from the blockRow startBlock.
      */
      static bool forwardSolve(   const agxSabre::SparseMatrix<double>& M,
                                  agxSabre::DenseVector<double>&  x,
                                  uint32_t len,
                                  uint32_t startBlock = 0 );

      /**
      Solve inplace M x_out = x_in  using M(1:len,1:len)

      Reduce the amount of work by having the sparsity of x stored in the queue
      and use the elimination tree to know where to go next.
      */
      static bool forwardSolve(   const agxSabre::SparseMatrix<double>& M,
                                  agxSabre::DenseVector<double>&  x,
                                  uint32_t len,
                                  agxSabre::UInt32Set& queue,
                                  const agxSabre::EliminationTree* etree );


      static bool forwardSolve(   const agxSabre::SparseMatrix<double>& M, double* x );
      static bool forwardSolve(   const agxSabre::SparseMatrix<double>& M, agxSabre::DenseVector<double>& x );

      static bool backwardTSolve( const agxSabre::SparseMatrix<double>& M, double* x );
      static bool backwardTSolve( const agxSabre::SparseMatrix<double>& M, agxSabre::DenseVector<double>& x );

      static bool diagSolve(      const agxSabre::DenseVector<double>& M,  double* x );
      static bool diagSolve(      const agxSabre::DenseVector<double>& M,  agxSabre::DenseVector<double>& x );
  };

}

inline bool agxSabre::SabreSolve::forwardSolve(  const agxSabre::SparseMatrix<double>& M, agxSabre::DenseVector<double>& x )
{
  return forwardSolve( M, x.getData() );
}


inline bool agxSabre::SabreSolve::backwardTSolve( const agxSabre::SparseMatrix<double>& M, agxSabre::DenseVector<double>& x )
{
  return backwardTSolve( M, x.getData() );
}

#endif


