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

#ifndef AGXSABRE_CHOLESKY_H
#define AGXSABRE_CHOLESKY_H

#include <cmath>

#include <agx/config.h>
#include <agx/macros.h>

#ifdef _MSC_VER
#  define NOMINMAX
#endif


#include <iostream>

//#define CHOLESKY

namespace agxSabre
{
  AGX_FORCE_INLINE float Tsqrt( float f ) { return sqrtf( f ); }
  AGX_FORCE_INLINE double Tsqrt( double f ) { return sqrt( f ); }

  /**
     Class which implements cholesky decomposition and solve.

     Works similar to the lapack methods {d,s}potr{f,s} but less
     versatile regarding where L is stored.
  */
  struct Cholesky
  {
    /**
       Performs inplace Cholesky decomposition on data.
       Data is assumed to be stored rowmajor in upper triangle.
    */

    template< typename T >
    static AGX_FORCE_INLINE bool factor( T* data, int lda, int n )
    {
      // upper triangle is stored
      // row major data

      int ilda;
      T ajj;

      for (int i = 0; i < n; ++i )
      {
        ilda = i * lda;

        ajj = data[i + ilda ];

#ifdef CHOLESKY
        // Cholesky
        // L(i,i) = A(i,i) - L(i,0:i-1) * L(i,0:i-1)
        for (int j = 0; j < i; ++j)
          ajj -= data[ j * lda + i ] * data[ j * lda + i ];
        ajj = sqrt(ajj);
#else
        // LDLT:
        // LDLT: L(i,i) = A(i,i) - L(i,0:i-1) * D(i,i) * L(i,0:i-1)
        for (int j = 0; j < i; ++j)
          ajj -= data[ j * lda + i ] * data[ j * lda + i ] * data[ j*lda + j];
#endif
        data[ ilda + i] = ajj;

        // handle rest of row I
        for (int j = i+1; j < n; ++j )
        {
#ifdef CHOLESKY
          // Cholesky:
          // L(i,j) -= L(i,k) * L(j,k)
          for (int k = 0; k < i; ++k)
            data[ ilda + j ] -= data[ k * lda + i] * data[ k*lda + j ];
#else
          // For LDLT
          // L(i,j) -= L(i,k) * D(k,k) L(j,k)
          for (int k = 0; k < i; ++k)
            data[ ilda + j ] -= data[ k * lda + i] * data[k*lda + k]* data[ k*lda + j ];
#endif

          // L(i,j) /= L(j,j)
          data[ ilda + j ] /= data[ ilda + i ];
        }
      }

      return true;
    }



    template< typename T >
    static AGX_FORCE_INLINE void solve( T* data, int lda, int n, T* rhs )
    {
      // have upper triangle stored (row major)
      // L * (L' * x) = rhs

      // 1. L y = rhs
      for (int i = 0 ; i < n; ++i)
      {
        for (int j = 0; j < i; ++j )
          rhs[i] -= rhs[j] * data[ j * lda + i ];
#ifdef CHOLESKY
        rhs[ i ] /= data[ i*lda + i];
#endif
      }

#ifndef CHOLESKY
      for (int i = 0; i< n; ++i )
        rhs[ i ] /= data[ i*lda + i];
#endif

      // 2. L' x = y
      for (int i = n-1 ; i >= 0; --i)
      {
        for (int j = n-1; j > i; --j )
          rhs[i] -= rhs[j] * data[i*lda + j ];
#ifdef CHOLESKY
        rhs[ i ] /= data[ i*lda + i];
#endif
      }

    }

  };




}

#endif

