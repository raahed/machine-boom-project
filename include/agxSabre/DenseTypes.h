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

#ifndef AGXSABRE_DENSETYPES_H
#define AGXSABRE_DENSETYPES_H

#include <agxSabre/agxSabre.h>

#include <agx/debug.h>
#include <cstdio>
#include <string.h>

namespace agxSabre
{

  template< typename T >
  class DenseMatrix
  {
    public:
      /**
      Default constructor.
      */
      DenseMatrix();

      /**
      Constructor, creates an empty DenseMatrix.
      \param i Number of rows
      \param j Number of columns
      */
      DenseMatrix( size_t i, size_t j );

      /**
      Constructor, creates a DenseMatrix from column compressed data.
      \param i    Number of rows
      \param j    Number of columns
      \param nnz  Number of elements in the compressed format
      \param r    Row index, contains nnz values.
      \param c    Compressed columns, contains j+1 values.
      \param data agxData::Values to be written into the matrix, contains nnz values.
      */
      DenseMatrix( size_t i, size_t j, size_t nnz, int* r, int* c, T* data );

      /**
      Copy Constructor
      */
      DenseMatrix( const DenseMatrix<T>& other );

      /**
      Destructor.
      */
      ~DenseMatrix();

      /**
      Assignment operator.
      */
      DenseMatrix& operator= ( const DenseMatrix& rhs );

      /**
      Clears the matrix, i.e, sets all elements to zero and
      state to unfactored.
      */
      void clear();

      /**
      Changes the size of the matrix, previous
      values in the matrix will be lost.
      */
      void setSize(size_t i, size_t j);

      /**
      \return Number of rows in the matrix.
      */
      size_t getNumRows() const;

      /**
      \return Number of columns in the matrix.
      */
      size_t getNumColumns() const;

      /**
      \return Returns a pointer to the matrix data which
              is written in row major format.
      */
      T* getData();

      /**
      \return Returns a pointer to the matrix data which
              is written in row major format.
      */
      const T* getData() const;

      /**
      Operator to access element in the matrix.
      */
      T operator() (size_t i, size_t j) const;

      /**
      Operator to access element in the matrix.
      */
      T& operator() (size_t i, size_t j);


      DenseMatrix operator* ( const DenseMatrix& other ) const;

      DenseMatrix transpose() const;

      /**
      Prints the matrix.
      */
      void print() const;


      /**
      Factors the matrix and overwrites previous values.
      Requires the matrix to be symmetric positive definite.
      */
      bool factor();


      /**
      Solves Ax=b.
      If the matrix isn't previously factored, factor will be called automatically.
      \param b Right hand side of the equation, b will be overwritten with the solution.
      */
      bool solve( T* b );


    private:
      size_t m_rows;
      size_t m_cols;
      T* m_data;
      bool m_factored;
  };


  template< typename T >
  class DenseVector
  {
    public:

      /**
      Default constructor.
      */
      DenseVector();

      /**
      Constructor.
      */
      DenseVector( size_t size );

      /**
      Copy constructor
      */
      DenseVector( const DenseVector<T>& other );

      /**
      */
      ~DenseVector();

      /**
      */
      DenseVector& operator= ( const DenseVector& rhs );

      /**
      */
      void clear();

      /**
      */
      void setSize(size_t size);

      /**
      */
      size_t getSize() const;

      /**
      */
      T* getData();

      /**
      */
      const T* getData() const;

      /**
      */
      T operator() (size_t i) const;
      T& operator() (size_t i);

      /**
      */
      T operator[] (size_t i) const;
      T& operator[] (size_t i);

      DenseVector<T> operator+( const DenseVector<T>& rhs );
      DenseVector<T> operator-( const DenseVector<T>& rhs );


      /**
      */
      void print() const;

    private:
      size_t m_size;
      size_t m_allocatedSize;
      T* m_data;
      T* m_mem;

      void allocateMemory( size_t size );
  };



  /////////////////////////////////////////////////////////////////////////////
  //
  // DenseMatrix

  template< typename T >
  DenseMatrix<T>::DenseMatrix()
    : m_rows(0), m_cols(0), m_data(0), m_factored( false )
  {
  }


  template< typename T >
  DenseMatrix<T>::DenseMatrix( size_t i, size_t j ) :
    m_rows(i), m_cols(j), m_data( new T[i*j]() ), m_factored( false )
  {
  }


  template< typename T >
  DenseMatrix<T>::DenseMatrix( size_t i, size_t j, size_t /* nnz */, int* r, int* c, T* data )
  {
    m_factored = false;

    m_rows = i;
    m_cols = j;
    m_data = new T[i*j]();

    int start,end;
    for (size_t v= 0; v < j; ++v )
    {
      start = c[v];
      end = c[v+1];
      for (int row = start; row < end; ++row )
        (*this)(r[row], v) = data[row];
    }
  }


  template< typename T >
  DenseMatrix<T>::DenseMatrix( const DenseMatrix<T>& other )
  {
    m_data = 0;
    *this = other;
  }


  template< typename T >
  DenseMatrix<T>::~DenseMatrix()
  {
    if (m_data)
      delete[] m_data;
    m_data = 0;
  }


  template< typename T >
  DenseMatrix<T>& DenseMatrix<T>::operator= ( const DenseMatrix& rhs )
  {
    if ( &rhs == this )
      return *this;

    if (m_data) delete[] m_data;
    m_cols = rhs.getNumColumns();
    m_rows = rhs.getNumRows();
    m_data = new T[m_rows*m_cols]();
    m_factored = rhs.m_factored;
    memcpy(m_data, rhs.m_data, m_rows*m_cols*sizeof(T) );
    return *this;
  }


  template< typename T >
  void DenseMatrix<T>::clear()
  {
    if (m_data)
      memset(m_data, 0, m_rows*m_cols*sizeof( T ) );
  }


  template< typename T >
  void DenseMatrix<T>::setSize(size_t i, size_t j)
  {
    if (m_data)
      delete[] m_data;
    m_rows = i;
    m_cols = j;
    m_data = new T[ m_rows * m_cols ]();
  }


  template< typename T >
  size_t DenseMatrix<T>::getNumRows() const
  {
    return m_rows;
  }


  template< typename T >
  size_t DenseMatrix<T>::getNumColumns() const
  {
    return m_cols;
  }


  template< typename T >
  T* DenseMatrix<T>::getData()
  {
    return m_data;
  }


  template< typename T >
  const T* DenseMatrix<T>::getData() const
  {
    return m_data;
  }


  template< typename T >
  T DenseMatrix<T>::operator() (size_t i, size_t j) const
  {
    agxAssert( i < m_rows && j < m_cols );
    return m_data[ j * m_rows + i ];
  }


  template< typename T >
  T& DenseMatrix<T>::operator() (size_t i, size_t j)
  {
    agxAssert( i < m_rows && j < m_cols );
    return m_data[ j * m_rows + i ];
  }

  template< typename T >
  DenseMatrix<T> DenseMatrix<T>::operator * ( const DenseMatrix<T>& other ) const
  {
    DenseMatrix<T> result = DenseMatrix<T>( this->m_rows, other.m_cols );

    if ( other.m_rows != m_cols )
      return result;

    for (size_t i = 0; i < m_rows; ++i)
      for ( size_t k = 0; k < other.m_cols; ++k )
        for( size_t j = 0; j < m_cols; ++j )
          result(i,k) += (*this)(i,j) * other(j,k);

    return result;
  }


  template< typename T >
  DenseMatrix<T> DenseMatrix<T>::transpose() const
  {
    DenseMatrix<T> result = DenseMatrix(m_cols, m_rows);

    for ( size_t i = 0; i < m_rows; ++i)
      for (size_t j = 0; j < m_cols; ++j )
        result(j,i) = (*this)(i,j);

    return result;
  }

  template< typename T >
  void DenseMatrix<T>::print() const
  {
    printf("DenseMatrix, size %ux%u\n", (unsigned int)m_rows, (unsigned int)m_cols );
//    printf("\t");
    for (size_t i = 0; i<m_rows; ++i)
    {
      for (size_t j = 0; j<m_cols; ++j)
      {
        printf("%7.2f ", (*this)(i,j) );
      }
      printf("\n");
    }
    printf("\n");
  }

/*
  void grow( size_t newSize )
  {
    newSize = std::max( newSize, std::max(m_rows, m_cols) );
    T* data = new T[ newSize*newSize ]();

    // copy old stuff
    for (size_t i=0; i<m_rows; ++i)
      for (size_t j=0; j<m_cols; ++j)
        data[i * newSize + j ] = m_data[i * m_rows + j];

    // fill new diagonal with 1:s
    for (size_t i=m_rows; i<newSize; ++i)
      data[i * newSize + i] = 1;

    delete[] m_data;
    m_data = data;
    m_rows = m_cols = newSize;
  }
*/


  template<>
  AGXSABRE_EXPORT bool DenseMatrix<double>::factor();

  template<>
  AGXSABRE_EXPORT bool DenseMatrix<double>::solve(double* b);


  template<>
  AGXSABRE_EXPORT bool DenseMatrix<float>::factor();

  template<>
  AGXSABRE_EXPORT bool DenseMatrix<float>::solve(float* b);





  /////////////////////////////////////////////////////////////////////////////
  //
  // DenseVector

  template< typename T >
  DenseVector<T>::DenseVector()
    : m_size(0), m_allocatedSize(0), m_data(0), m_mem(0)
  {
  }

  template< typename T >
  DenseVector<T>::DenseVector( size_t size )
    : m_size( size ), m_allocatedSize(size), m_data( 0 ), m_mem(0)
  {
    allocateMemory( size );
  }


  template< typename T >
  DenseVector<T>::DenseVector( const DenseVector<T>& other )
  {
    m_mem = 0;
    m_allocatedSize = 0;
    *this = other;
  }

  template< typename T >
  DenseVector<T>::~DenseVector()
  {
    if (m_mem)
      delete[] m_mem;
    m_mem = 0;
    m_data = 0;
  }


  template< typename T >
  void DenseVector<T>::allocateMemory( size_t size )
  {
    if ( m_mem )
      delete[] m_mem;

    m_allocatedSize = size;

    // allocate some extra elements so address can be aligned
    m_mem  = new T[ m_allocatedSize + (32/sizeof(T) -1) ]();
    m_data = agxSabre::alignPtr( m_mem );
  }

  template< typename T >
  DenseVector<T>& DenseVector<T>::operator= ( const DenseVector& rhs )
  {
    if ( &rhs == this )
      return *this;

    if ( m_allocatedSize < rhs.m_size )
      allocateMemory( rhs.m_size );

    m_size = rhs.getSize();
    memcpy(m_data, rhs.m_data, m_size*sizeof(T) );

    return *this;
  }

  template< typename T >
  void DenseVector<T>::clear()
  {
    m_size = 0;
    //if (m_data)
    //  memset(m_data, 0, m_size * sizeof(T) );
  }

  template< typename T >
  void DenseVector<T>::setSize(size_t size)
  {

    if ( size > m_allocatedSize )
      allocateMemory( size );

    m_size = size;
  }

  template< typename T >
  size_t DenseVector<T>::getSize() const
  {
    return m_size;
  }

  template< typename T >
  T* DenseVector<T>:: getData()
  {
    return m_data;
  }

  template< typename T >
  const T* DenseVector<T>::getData() const
  {
    return m_data;
  }

  template< typename T >
  T DenseVector<T>::operator() (size_t i) const
  {
    agxAssert( i < m_allocatedSize );
    return m_data[ i ];
  }

  template< typename T >
  T& DenseVector<T>::operator() (size_t i)
  {
    agxAssert( i < m_allocatedSize );
    return m_data[ i ];
  }

  template< typename T >
  T DenseVector<T>::operator[] (size_t i) const
  {
    agxAssert( i < m_allocatedSize );
    return m_data[ i ];
  }

  template< typename T >
  T& DenseVector<T>::operator[] (size_t i)
  {
    agxAssert( i < m_allocatedSize );
    return m_data[ i ];
  }

  template< typename T >
  void DenseVector<T>::print() const
  {
    printf("DenseVector, size %u\n", (unsigned int)m_size );
    printf("\t");
    for (size_t i = 0; i<m_size; ++i)
    {
      printf("%-7.2f ", (*this)(i) );
    }
    printf("\n\n");
  }

/*
  void grow( size_t newSize )
  {
    T* data = new T[ std::max(newSize,m_size) ];

    memcpy(data,m_data, m_size*sizeof(T));

    delete[] m_data;
    m_data = data;
    m_size = std::max(newSize,m_size);
  }
*/

  template< typename T >
  DenseVector<T> DenseVector<T>::operator+( const DenseVector<T>& rhs )
  {
    DenseVector<T> result;
    result = *this;
    for (size_t i=0; i < m_size; ++i)
      result.m_data[i] += rhs.m_data[i];
    return result;
  }

  template< typename T >
  DenseVector<T> DenseVector<T>::operator-( const DenseVector<T>& rhs )
  {
    DenseVector<T> result;
    result = *this;
    for (size_t i=0; i < m_size; ++i)
      result.m_data[i] -= rhs.m_data[i];
    return result;
  }


  template< typename T>
  DenseVector<T> operator*( const DenseMatrix<T>& m, const DenseVector<T>& v )
  {
    DenseVector<T> result;

    if ( m.getNumColumns() != v.getSize() )
      return result;

    result.setSize( v.getSize() );

    for ( size_t c = 0; c < m.getNumColumns(); ++c )
      for ( size_t r = 0; r < m.getNumRows(); ++r )
        result(r) += m(r,c) * v(c);

    return  result;
  }


}

#endif // AGXSABRE_DENSETYPES_H
