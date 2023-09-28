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

#ifndef SABRE_SKELETON_MATRIX
#define SABRE_SKELETON_MATRIX

#include <agxSabre/export.h>
#include <agxSabre/agxSabre.h>
#include <agxSabre/Permutation.h>
#include <agxSabre/ScratchPad.h>

namespace agxSabre
{

  /**
  When using Sabre, this is the first matrix being created.

  It's used to compute the permutation that should be used
  and then the layout of the blocks.
  */
  class SkeletonMatrix
  {
    public:

      /**
      Default constructor.
      */
      inline SkeletonMatrix();

      /**
      Constructor.
      \param n Size of matrix.
      \param pointers Pointers for packed format, must be n+1 elements.
      \param indices Column indices, must be pointers[n] elements.
      \param dimensions If non-null pointer, blocksize for each row.
      */
      inline SkeletonMatrix( size_t n, unsigned int* pointers, unsigned int* indices, unsigned int* dimensions = 0 );

      SkeletonMatrix( const SkeletonMatrix& other );

      inline ~SkeletonMatrix();

      SkeletonMatrix& operator= (const SkeletonMatrix& rhs);

      inline void clear();

      /**
      Sets the skeleton matrix from a row packed lower triangle.
      */
      inline void set( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions = 0 );

      /**
      \return Number of rows in the matrix
      */
      inline size_t getSize() const;


      /**
      \return Number of non zero elements in the matrix.
      */
      inline size_t getNumValues() const;


      /**
      \return The dimension for row r
      */
      inline unsigned int getDimension( size_t r ) const;


      /**
      \return Index where row r starts
      */
      inline unsigned int getRowPointer( size_t r ) const;


      /**
      \return Column index for element p
      */
      inline unsigned int getColumnIndex( size_t p ) const;


      /**
      \return If element p should be written transposed
      */
      inline bool getTranspose( size_t p ) const;


      /**
      Permutes the matrix according to perm
      */
      inline void permute( const SkeletonMatrix& other, const Permutation& perm, ScratchPad& pad );

      /**
      Prints the matrix structure
      */
      void print() const;


      /**
      Raw access to the dimension vector
      */
      inline const unsigned int* getDimensionVector() const;

      /**
      Raw access to rowpointers
      */
      inline const unsigned int* getPointerVector() const;

      /**
      Raw access to indices
      */
      inline const unsigned int* getIndexVector() const;

      /**
      The skeleton matrix holds the lower triangle of a symmetric matrix.

      Transpose will make it hold the upper triangle instead.
      This could be seen as holding the lower triangle stored by column
      instead.
      */
      inline SkeletonMatrix transpose() const;


      /**
      This method will transpose a compressed matrix format and store the
      result in the current SkeletonMatrix.

      Resulting matrix will not contain:
       - transpose flags for blocks
       - dimension for blocks
      */
      inline void transpose( int n, const int* pointers, const int* indices, ScratchPad& pad );

    private:
      unsigned int *m_columnIndices;  // #m_nnz
      unsigned int *m_rowPointers;    // #m_n+1

      unsigned int *m_dimension;      // #m_n

      bool *m_transpose;     // #m_nnz

      size_t m_n;

      size_t m_allocatedRowSize;
      size_t m_allocatedElementSize;


      inline void allocateMemory( size_t n, size_t nnz );
      inline void releaseMemory();
  };


  inline SkeletonMatrix::SkeletonMatrix() :
    m_columnIndices(0), m_rowPointers(0), m_dimension(0), m_transpose(0), m_n(0), m_allocatedRowSize(0), m_allocatedElementSize(0)
  {
  }

  inline SkeletonMatrix::SkeletonMatrix(size_t n, unsigned int* pointers, unsigned int* indices, unsigned int* dimensions ) :
    m_columnIndices(0), m_rowPointers(0), m_dimension(0), m_transpose(0), m_n(0), m_allocatedRowSize(0), m_allocatedElementSize(0)
  {
    set(n, pointers, indices, dimensions );
  }


  inline SkeletonMatrix::SkeletonMatrix( const SkeletonMatrix& other ) :
    m_columnIndices(0), m_rowPointers(0), m_dimension(0), m_transpose(0),
    m_n(0), m_allocatedRowSize(0), m_allocatedElementSize(0)
  {
    *this = other;
  }

  inline SkeletonMatrix::~SkeletonMatrix()
  {
    releaseMemory();
  }


  inline SkeletonMatrix& SkeletonMatrix::operator= (const SkeletonMatrix& rhs)
  {
    if ( this == &rhs )
      return *this;

    this->set( rhs.getSize(), rhs.getPointerVector(), rhs.getIndexVector(), rhs.getDimensionVector() );
    memcpy( m_transpose, rhs.m_transpose, sizeof(bool) * m_rowPointers[ m_n ] );
    return *this;
  }


  inline void SkeletonMatrix::clear()
  {
    m_n = 0;
  }


  inline void SkeletonMatrix::set( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions )
  {
    m_n = n;

    if ( m_n > m_allocatedRowSize || pointers[m_n] > m_allocatedElementSize )
    {
      releaseMemory();
      allocateMemory( m_n, pointers[m_n] );
    }

    memcpy( m_rowPointers, pointers, (m_n+1)*sizeof(unsigned int) );
    memcpy( m_columnIndices, indices, (m_rowPointers[m_n])*sizeof(unsigned int) );

    if (dimensions)
      memcpy( m_dimension, dimensions, (m_n)*sizeof(unsigned int) );

    memset( m_transpose, false, m_allocatedElementSize * sizeof(bool) );
  }



  inline size_t SkeletonMatrix::getSize() const
  {
    return m_n;
  }


  inline size_t SkeletonMatrix::getNumValues() const
  {
    return (m_n != 0 ? m_rowPointers[m_n] : 0 );
  }


  inline unsigned int SkeletonMatrix::getDimension( size_t r ) const
  {
    return m_dimension[r];
  }


  inline unsigned int SkeletonMatrix::getRowPointer( size_t r ) const
  {
    return m_rowPointers[r];
  }


  inline unsigned int SkeletonMatrix::getColumnIndex( size_t p ) const
  {
    return m_columnIndices[p];
  }


  inline bool SkeletonMatrix::getTranspose( size_t p ) const
  {
    return m_transpose[p];
  }


  inline void SkeletonMatrix::permute( const SkeletonMatrix& other, const Permutation& perm, ScratchPad& pad )
  {
    set( other.m_n, other.m_rowPointers, other.m_columnIndices, other.m_dimension );

    size_t tag = 0;

    int* work = pad.getIntArray( tag, m_n );

    memset( m_rowPointers, 0, sizeof(int) * m_n );
    memset( work, 0, sizeof(int) * m_n );

    //
    // calculate number of elements on each row in the
    // permuted matrix.
    //
    for (size_t i=0; i<m_n; ++i)
    {
      // on row i in the permuted matrix, write row I.
      const int I = perm.permute( (int) i);

      for (unsigned int p = other.m_rowPointers[I]; p < other.m_rowPointers[I+1]; ++p)
      {
        const int J = m_columnIndices[p];
        const size_t j = (size_t) perm.inversePermute(J);
        work[std::max(i, j)] ++ ;
      }
    }

    //
    // Setup row pointers for the permuted matrix.
    //
    m_rowPointers[0] = 0;
    int temp = 0;
    for (size_t i=0; i<m_n; )
    {
      temp += work[i];
      m_rowPointers[++i] = temp;
    }


    //
    // clear data before putting things at the correct places
    //
    memset( m_columnIndices, 0, sizeof(int)* m_rowPointers[m_n] );
    memset( m_transpose, 0, sizeof(bool)* m_rowPointers[m_n] );
    memset( work, 0, sizeof(int)*m_n );

    //
    // Rebuild skeleton matrix by putting things at the right place
    //
    for (size_t i=0; i<m_n; ++i)
    {
      // on row i, write row I
      const int I = perm.permute( (int) i);

      // loop over row I
      for (unsigned int p= other.m_rowPointers[I]; p < other.m_rowPointers[I+1]; ++p)
      {
        const int J = other.m_columnIndices[p];
        const size_t j = (size_t) perm.inversePermute(J);

        /// invert the permutation flag as needed, i.e., if (i0, j0) = (i, j).
        const int i0 = (int) std::max(i, j);
        const int j0 = (int) std::min(i, j);

        const unsigned int address = m_rowPointers[i0] + work[i0]++;

        m_columnIndices[ address ] = j0;

        // Non diagonal element,
        // should be transposed if we bounced of the diagonal (j > i)
        m_transpose[address] = ( j > i );


        // C->m_data[address].m_data  = A.m_data[p].m_data;


        /**
          Now, make sure that insertion along rows proceeds in order.
          Observe that any previous insertion on this row had to be in
          order since it came from data past the diagonal for a row
          i1<i0, and this ended up in colum j1 = i1.  Thus, as we
          proceed along the rows, everytime we transpose, we have a j1
          bigger than preceeding ones.   Therefore, just checking the
          last element  for the case where i0 == i is sufficient.

          However, it is still necessary to do a bubble sort with each
          new element added and this can be catastrophic.
         */

        // if we're not the first element on the row, make sure we get to the right position
        //
        if ( address > m_rowPointers[i0] )
        {
          int a0 = address;    // index of element to insert
          int a1 = address -1; // index of element to compare with

          while ( i0== (int)i &&
                  a1 >= (int)m_rowPointers[i0] &&                   // don't walk into previous row
                  m_columnIndices[a1] > m_columnIndices[a0] // columns out of order
                )
          {
            std::swap( m_transpose[a1], m_transpose[a0] );
            std::swap( m_columnIndices[a1], m_columnIndices[a0] );

            //std::swap(C->m_cidx[a1], C->m_cidx[a0]);
            //std::swap(C->m_data[a1], C->m_data[a0]);
            a0 -= 1;
            a1 -= 1;
          }
        }
      }
    }

    //
    // update dimension vector
    //
    for ( size_t i=0; i<m_n; ++i )
      work[i] = m_dimension[ perm.permute( (int) i) ];

    for ( size_t i=0; i<m_n; ++i )
      m_dimension[i] = work[i];

  }



  inline void SkeletonMatrix::allocateMemory( size_t n, size_t nnz )
  {
    m_allocatedRowSize = n;

    m_rowPointers = new unsigned int[ n+1 ];
    m_dimension = new unsigned int[ n ];

    m_allocatedElementSize = nnz;

    m_columnIndices = new unsigned int[ nnz ];
    m_transpose = new bool[ nnz ];
  }


  inline void SkeletonMatrix::releaseMemory()
  {
    if ( m_rowPointers )
      delete[] m_rowPointers;
    m_rowPointers = 0;

    if ( m_dimension )
      delete[] m_dimension;
    m_dimension = 0;

    if ( m_columnIndices )
      delete[] m_columnIndices;
    m_columnIndices = 0;

    if ( m_transpose )
      delete[] m_transpose;
    m_transpose = 0;
  }



  inline const unsigned int* SkeletonMatrix::getDimensionVector() const
  {
    return m_dimension;
  }

  inline const unsigned int* SkeletonMatrix::getPointerVector() const
  {
    return m_rowPointers;
  }

  inline const unsigned int* SkeletonMatrix::getIndexVector() const
  {
    return m_columnIndices;
  }



  inline SkeletonMatrix SkeletonMatrix::transpose() const
  {
    if ( m_n < 1 )
      return SkeletonMatrix();

    SkeletonMatrix result;
    result.allocateMemory( m_n, m_allocatedElementSize );
    result.m_n = m_n;

    int* work = new int[ m_n ];
    memset( work, 0, sizeof(int) * m_n );

    // count number of elements in each column
    for (unsigned int p=0; p<m_rowPointers[m_n]; ++p)
    {
      const int j = m_columnIndices[p];
      ++work[ j ];
    }


    result.m_rowPointers[0] = 0;

    // setup new rowpointers
    for ( unsigned int r = 0; r < m_n; ++r )
    {
      result.m_rowPointers[r+1] = result.m_rowPointers[r] + work[r];
      work[r] = 0;
    }

    // construct transpose
    for ( unsigned int r = 0; r < m_n; ++r )
    {

      result.m_dimension[r] = m_dimension[r];

      for ( unsigned int p = m_rowPointers[r]; p < m_rowPointers[r+1]; ++p )
      {
        int c = m_columnIndices[p];
        // append new element on row 'c', at column 'r' in result

        int idx = work[c] + result.m_rowPointers[c];
        work[c]++;

        result.m_columnIndices[idx] = r;
        result.m_transpose[idx]     = m_transpose[p];
      }
    }

    delete[] work;
    return result;
  }




  inline void SkeletonMatrix::transpose( int n, const int* pointers, const int* indices, ScratchPad& pad )
  {
    if ( n < 1 )
      return;

    m_n = n;

    if ( m_n > m_allocatedRowSize || pointers[m_n] > (int)m_allocatedElementSize )
    {
      releaseMemory();
      allocateMemory( m_n, pointers[m_n] );
    }

    size_t tag = 0;

    int* work = pad.getIntArray( tag, n );
    memset( work, 0, sizeof(int) * n );

    // count number of elements in each column
    for (int p=0; p<pointers[n]; ++p)
    {
      const int j = indices[p];
      ++work[ j ];
    }

    m_rowPointers[0] = 0;

    // setup new rowpointers
    for ( int r = 0; r < n; ++r )
    {
      m_rowPointers[r+1] = m_rowPointers[r] + work[r];
      work[r] = 0;
    }

    // construct transpose
    for ( int r = 0; r < n; ++r )
    {
      for ( int p = pointers[r]; p < pointers[r+1]; ++p )
      {
        int c = indices[p];
        // append new element on row 'c', at column 'r' in result

        int idx = work[c] + m_rowPointers[c];
        work[c]++;

        m_columnIndices[idx] = r;
      }
    }
  }


}

#endif

