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

#ifndef SABRE_PERMUTATION_H
#define SABRE_PERMUTATION_H

#include <cstdlib>
#include <cstring>

#include <agxSabre/export.h>

namespace agxSabre
{

  /**
  Permutation holds a symmetric reordering of the matrix.

  For some matrix A, the permutation forms a new matrix Ap = A(p,p)
  The inverse permutation Ap(Pinv,Pinv) gives A.

  permute( inversePermute( i ) ) == i.
  */
  class Permutation
  {
    public:

      /**
      Constructor
      */
      inline Permutation( size_t n = 0 );

      inline ~Permutation();

      Permutation& operator= (const Permutation& rhs );

      /**
      Clears the permutation, i.e. p(i) = i.
      */
      inline void clear();


      inline size_t getSize() const;
      inline void setSize( size_t n );


      inline int* getPermutation();
      inline int* getInversePermutation();

      inline const int* getPermutation() const;
      inline const int* getInversePermutation() const;

      inline int permute( int i ) const;

      inline int inversePermute( int i ) const;

      inline void postOrder( const int* p );

      /**
      Computes the inverse permutation.
      */
      inline void computeInversePermutation();

      /**
      Prints the permutation
      */
      void print() const;


      /**
      Merges this permutation, p, to: outer( p( i ) )
      */
      inline void merge( const Permutation& outer );



    private:
      size_t m_n;
      size_t m_allocatedSize;

      // int so colamd/symamd can write directly into local buffer
      int* m_perm;
      int* m_invPerm;

      Permutation( const Permutation& /*other*/ ) : m_n(0), m_allocatedSize(0), m_perm(0), m_invPerm(0) {}

      /**
      \param n Number of elements to allocate
      */
      inline void allocateMemory(size_t n);
      inline void releaseMemory();
  };


  inline Permutation::Permutation( size_t n ) :
    m_n(n), m_allocatedSize(0), m_perm(0), m_invPerm(0)
  {
    setSize( m_n );
  }


  inline Permutation::~Permutation()
  {
    releaseMemory();
  }

  inline Permutation& Permutation::operator= (const Permutation& rhs )
  {
    if ( &rhs == this )
      return *this;

    this->setSize( rhs.getSize() );
    memcpy( this->m_perm, rhs.m_perm, sizeof(int) * rhs.getSize() );
    memcpy( this->m_invPerm, rhs.m_invPerm, sizeof(int) * rhs.getSize() );

    return *this;
  }

  inline void Permutation::clear()
  {
    if ( m_perm )
      for (size_t i = 0; i<m_allocatedSize; ++i)
        m_perm[i] = (int)i;

    if ( m_invPerm )
      for (size_t i = 0; i<m_allocatedSize; ++i)
        m_invPerm[i] = (int)i;

  }

  inline size_t Permutation::getSize() const
  {
    return m_n;
  }

  inline void Permutation::setSize( size_t n )
  {
    m_n = n;

    if ( m_n > m_allocatedSize )
    {
      releaseMemory();
      allocateMemory(n);
    }
  }

  inline int* Permutation::getPermutation()
  {
    return m_perm;
  }

  inline int* Permutation::getInversePermutation()
  {
    return m_invPerm;
  }

  inline const int* Permutation::getPermutation() const
  {
    return m_perm;
  }

  inline const int* Permutation::getInversePermutation() const
  {
    return m_invPerm;
  }




  inline int Permutation::permute( int i ) const
  {
    return m_perm[i];
  }


  inline int Permutation::inversePermute( int i ) const
  {
    return m_invPerm[i];
  }



  inline void Permutation::computeInversePermutation()
  {
    for (size_t i=0; i<m_n; ++i)
      m_invPerm[ m_perm[i] ] = (int)i;

  }


  inline void Permutation::allocateMemory(size_t n)
  {
    // the one extra element is needed by colamd,
    // it uses the buffer as workarea
    m_perm = new int[n+1];
    m_invPerm = new int[n+1];

    m_allocatedSize = n;
  }


  inline void Permutation::releaseMemory()
  {
    if (m_perm)
      delete[] m_perm;
    m_perm = 0;

    if (m_invPerm)
      delete[] m_invPerm;
    m_invPerm = 0;

    m_allocatedSize = 0;
  }


  inline void Permutation::merge( const Permutation& outer )
  {
    for (size_t i = 0; i < m_n; ++i)
      m_perm[i] = outer.permute( m_perm[i] );
  }


  inline void Permutation::postOrder( const int* p )
  {
    for ( size_t i = 0; i < m_n; ++i )
      m_invPerm[ p[i] ] = m_perm[i];

    memcpy( m_perm, m_invPerm, m_n * sizeof( int ) );
    //for ( size_t i = 0; i < m_n; ++i )
    //  m_perm[i] = m_invPerm[i];

    for ( size_t i = 0; i < m_n; ++i )
      m_invPerm[ m_perm[i] ] = (int)i;
  }

}


#endif

