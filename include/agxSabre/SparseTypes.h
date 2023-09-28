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

#ifndef AGXSABRE_SPARSETYPES_H
#define AGXSABRE_SPARSETYPES_H


#include <cfloat>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <cassert>

#include <agxSabre/agxSabre.h>
#include <agxSabre/DenseTypes.h>
#include <agxSabre/MetaLayout.h>

namespace agxSabre
{



  // fwd declaration
  template <typename T >
  class SparseMatrix;

  class EliminationTree;


  /**
  A data block is a view into data. A DataBlock is not responsible for
  allocating and freeing the memory. Data blocks are stored in row major order.
  */
  template< typename T >
  class DataBlock
  {
    public:
      /**
      Default constructor.
      */
      DataBlock();

      /**
      Constructor.
      \param data Pointer to data.
      \param rows Number of rows in block.
      \param cols Number of columns in block.
      */
      DataBlock( T* data, size_t rows, size_t cols, size_t lda );

      /**
      Assignment operator
      */
      DataBlock<T>& operator=( const DataBlock<T>& rhs );

      /**
      Access element in block with block local row and column indices.

      No checks are performed to verify that 0 <= i < m_rows and
      0 <= j < m_cols.
      */
      T& operator()(size_t i, size_t j);

      /**
      Access element in block with block local row and column indices
      */
      const T& operator()(size_t i, size_t j) const;

      /**
      \return Pointer to the actual data
      */
      T* getPointer();

      /**
      \return Pointer to the actual data
      */
      const T* getPointer() const;


      /**
      Changes the data pointer and size for the block.
      */
      void set( T* ptr, size_t rows, size_t cols, size_t lda );

      /**
      Changes the data pointer for the block
      */
      void setPointer( T* ptr );

      /**
      Changes the logical view of how the contents of the block is accessed.
      \param rows How many rows that are in the block
      \param cols How many columns that are in the block
      */
      void setSize( size_t rows, size_t cols, size_t lda );



      const size_t& getNumRows() const;
      const size_t& getNumColumns() const;

      /**
      Prints the datablock
      */
      void print() const;

      /**
      Copies the data from another block with the same size.
      No checks are performed about block sizes
      \param other Block contents to be copied into this block
      */
      void copy( const DataBlock<T>& other );

      /**
      Copies data from an address into the block. No checks are performed
      and the data is assumed to be stored in row major order.
      */
      void copy( const T* );

      /**
      Copies the diagonal to ptr
      \param ptr Where to store the diagonal
      */
      void getDiagonal( T* ptr ) const;

      /**
      Copies the columns from a pointer to data (belonging to a block with same memory layout)
      to this block and scales the stored values.
      */
      void copyScaledColumns( const T* blockPtr, const T* scaleFactors );

      /**
      */
      void copyInvScaledColumns( const T* blockPtr, const T* scaleFactors );

      /**
      Changes current block: column(i) / scaleFactors(i)
      */
      void invScaleColumns( const T* scaleFactors );

      /**
      Clears this datablock, i.e. fills it with zeros
      */
      void clear();

      DataBlock<T>& operator+=( const DataBlock<T>& other );

    private:
      T* m_data;
      size_t m_cols;
      size_t m_rows;
      size_t m_lda;
  };




  /**
    A sparse row is a view into a SparseMatrix.

  */
  template< typename T >
  class SparseRow
  {
    public:
      SparseRow( SparseMatrix<T>* matrix, size_t row);

      /**
      */
      void print() const;

      /**
      */
      void printStructure() const;

      /**
      */
      size_t length() const;

      /**
      */
      size_t getNumBlocks() const;

      /**
      If there is a block in column j, fill block
      with data access information.
      \param column Column block index
      \param block Block to be filled
      \return true if block found and information stored
      */
      bool operator()( size_t column, DataBlock<T>& block );

      /**
      */
      T* getBlockPointer( size_t column );

      /**
      \return Pointer to blockcolumn if block exists
      */
      const T* getBlockPointer( size_t column ) const;

      size_t getFirstBlockColumnIndex() const;

    private:
      class SparseMatrix<T>* m_matrix; // ptr to matrix
      size_t m_rowIndex;               // block row index
  };




  /**
  A SparseMatrix holding blocks of data.


  */
  template< typename T >
  class SparseMatrix
  {
    public:

      /**
      SparseMatrix constructor.
      \param blockSize number of columns for each block
      */
      SparseMatrix();


      /**
      SparseMatrix constructor.
      \param blockSize Size of blocks
      \param i number of rows
      \param j number of cols (i must be equal to j)
      \param nnz Number of non-zero elements to put into matrix
      \param r Row indices
      \param c Compressed column indices
      \param data agxData::Values
      */
      SparseMatrix( size_t blockSize, size_t i, size_t j, size_t nnz, int* r, int* c, double* data );



      /**
      SparseMatrix constructor.
      \param blockSizeVector vector with size for each block row
      \param numBlockRows number of block rows
      \param i number of rows
      \param j number of cols (i must be equal to j)
      \param nnz Number of non-zero elements to put into matrix
      \param r Row indices
      \param c Compressed column indices
      \param data agxData::Values
      */
      SparseMatrix( size_t* blockSizeVector, size_t numBlockRows, size_t i, size_t j, size_t nnz, int* r, int* c, double* data );


      /**
      Utility method for SparseMatrices during debug.
      */
      SparseMatrix( const DenseMatrix<T>& d, size_t blockSize );


      /**
      SparseMatrix destructor.
      */
      ~SparseMatrix();



      /**
      */
      DenseMatrix<T> getAsDenseMatrix() const;

      /**
      */
      SparseRow<T> row( size_t index );

      /**
      */
      const SparseRow<T> row( size_t index ) const;


      /**
      \return The index for the first block for row r
      */
      const unsigned int& getRowPointer( size_t r ) const;

      /**
      */
      const unsigned int* getPointerVector() const;

      /**
      */
      const unsigned int* getIndexVector() const;


      T* getBlockPointer( const size_t& blockIndex );
      const T* getBlockPointer( const size_t& blockIndex ) const;

      /**
      \return The column index for block number blockIndex
      */
      const unsigned int& getColumnIndex( const size_t& blockIndex ) const;


      /**
      \return Number of blocks for row r
      */
      size_t getNumBlocksInRow( size_t r ) const;

      /**
      \return Number of blocks currently in matrix
      */
      size_t getNumBlocks() const;

      /**
      \return The flag for block blockIndex
      */
      const unsigned int& getFlag( const size_t& blockIndex ) const;

      /**
      Sets the flag for block blockIndex
      */
      void setFlag( const size_t& blockIndex, unsigned int flag );

      /**
      Clears the matrix
      */
      void clear();


      /**
      Specifies number of block rows and block columns for the matrix.

      \param rows Number of block rows.
      \param cols Number of block columns.
      \param blocks Number of blocks
      \param elems Number of elements in all blocks including padding. Used to avoid memory allocations.
      */
      void setSize( size_t rows, size_t cols, size_t blocks, size_t elems );


      /**
      Acquires a block. The data in the block should be kept in row major order.

      Before calling this method, setSize must have been called.

      When calling this method to setup the matrix, the calls should do blockRowIndex
      in an increasing order and each block row must be completed before starting on the next one.
      Additionally, the blockColIndex should appear in an increasing order for each block row.

      Empty blockrows without any blocks are not supported.

      \param startRow Row number in matrix where block starts
      \param startCol Column number in matrix where block starts
      \param numRows  Number of rows in the block
      \param numCols  Number of columns in the block
      \param flag     Flag for the block
       */
      T* acquireBlock( size_t blockRowIndex, size_t blockColIndex, size_t numRows, size_t numCols, unsigned int flag = 0 );


      /**
      Setup the entire matrix block structure. Data allocations and size must be known in advance
      via call to setSize.

      \param rowPointers   Compressed info about rows
      \param colIndices    Column indices
      \param rowDimensions Size for each blockrow
      \param layoutByRow   Determines how the internal block data should be handled.

      */
      void batchConfigureMatrixLayout( const int* rowPointers,
                                       const int* colIndices,
                                       const unsigned int* rowDimensions,
                                       bool layoutByRow = true );


      /**
      Returns number of bytes used for the matrix
      */
      size_t getMemoryUsage() const;


      // debug
      void print() const;
      void printStructure() const;


      /**
      \return Number of rows in blockRow
      */
      AGX_FORCE_INLINE  const unsigned int& getBlockRowDimension( size_t blockRow ) const;


      /**
      \return Pointer to data about number or rows in blockrows
      */
      AGX_FORCE_INLINE const unsigned int* getBlockRowDimensions() const;



      /**
      \return Number of rows in blockRow including padding.
      */
      AGX_FORCE_INLINE  unsigned int getBlockRowPaddedDimension( size_t blockRow ) const;


      /**
      \return Pointer to data about padded number of rows in blockrows
      */
      AGX_FORCE_INLINE  const unsigned int* getBlockRowPaddedDimensions() const;



      /**
      \return Where in the big matrix the blockRow is located
      */
      AGX_FORCE_INLINE  const unsigned int& getBlockRowStartRow( size_t blockRow ) const;


      /**
      */
      size_t getMaxBlockRowDimension() const;

      /**
      \return Number of block rows for the matrix
      */
      AGX_FORCE_INLINE  size_t getNumBlockRows() const;

      /**
      \return Number of block columns
      */
      AGX_FORCE_INLINE  size_t getNumBlockColumns() const;

      /**
      \return Number of rows in the matrix (sum of block row dimensions)
      */
      size_t getNumRows() const;


      /**
      Calculates r = A x

      Matrix A is assumed to be triangular and symmetric.
      */
      void multiply( const T* x, T* r ) const;

      /**
      Calculates r = A x + r

      Matrix A is assumed to be triangular and symmetric.
      */
      void multiplyAdd( const T* x, T* r ) const;

      /**
      Calculates r = alpha A x + beta r

      Matrix A is assumed to be triangular and symmetric.
      */
      void multiplyAdd( const T& alpha, const T* x, const T& beta, T* r ) const;


      /**
      \return true if matrix has a structure for column traversal.
      A newly created matrix will not have this built.
      \see buildAdditionalColumnLayout
      */
      bool haveAdditionalColumnLayout() const;

      /**
      Build construct a structure for fast column traversal.
      */
      void buildAdditionalColumnLayout();


      /**
      Return the object representing the column layout
      */
      const MetaLayout_t& getColumnLayout() const;



    private:
      /**
      No copy constructor.
      */
      SparseMatrix( const SparseMatrix<T>& ) { abort(); }


      /**
      No assignment operator
      */
      SparseMatrix<T>& operator= ( const SparseMatrix<T>& ) { abort(); return *this; }


      friend class SparseRow<T>;

      T* m_realData;                  // Allocated data
      T* m_data;                      // Aligned data

      unsigned int* m_columnIndices;  // Which column blk belongs to  (#blks)

      unsigned int* m_flags;          // flags for each block, bits:  (#blks)

      T** m_blockPointers;            // Pointer into actual data     (#blks)

      unsigned int* m_rowPointers;    // Index in columnIndices for each row. Can
                                      // also be used to count number of blocks
                                      // for each row (#blockrows + 1)

      unsigned int* m_dimensions;     // Number of rows in each block row
                                      // (#blockrows + 1)

      unsigned int* m_dimensionSum;   // Where each block row starts in the big matrix

      unsigned int* m_paddedDimensions;

      size_t m_numBlocks;             // used number of blocks


      size_t m_numBlockRows;          // number of block rows
      size_t m_numBlockColumns;       // number of block columns

      size_t m_allocatedDataElements; //

      size_t m_allocatedBlockRows;    //
      size_t m_allocatedBlocks;       //

      MetaLayout_t m_columnLayout;    //

      /**
      Frees the memory used by the matrix.
      */
      void releaseMemory();

      void clearStructures();

  };


  /////////////////////////////////////////////////////////////////////////////
  //
  // SparseMatrix

  template< typename T >
  SparseMatrix<T>::SparseMatrix()
  {
    clearStructures();
  }


  // Construct sparse matrix from column compressed data
  // (assuming matrix is symmetric and entire matrix is stored)
  // Fixed block size version
  template< typename T >
  SparseMatrix<T>::SparseMatrix( size_t blockSize, size_t i, size_t j, size_t /*nnz*/, int* r, int* c, double* data )
  {
    clearStructures();

    DataBlock<T> block( 0, blockSize, blockSize, blockSize );


    if ( blockSize == 0 ||  i % blockSize != 0 || j % blockSize != 0 || i != j )
    {
      std::cerr << "Could not fill sparse matrix with column compressed data" << std::endl;
    }
    else
    {
      int numBlocks = 0;

      m_numBlockRows = i / blockSize;
      m_numBlockColumns = j / blockSize;

      //
      // traverse matrix, count number of blocks
      //
      for ( size_t blockCol = 0; blockCol < getNumBlockRows(); ++blockCol )
      {
        for ( size_t blockRow = 0; blockRow <= blockCol; ++blockRow )
        {
          // block is empty
          bool haveData = false;

          // loop over rows and columns in actual block
          // slightly lazy, loop over entire column to see if there is some data
          for ( size_t col = blockCol * blockSize; col < (blockCol+1)*blockSize; ++col )
            for ( size_t row = c[col]; row < (size_t)c[col+1]; ++row )
              haveData |= (( (size_t)r[row] >= blockRow * blockSize ) && ((size_t)r[row] < (blockRow+1)*blockSize ));

          if ( haveData )
            ++numBlocks;
        }
      }

      //
      // setup matrix, allocate memory
      //
      setSize(m_numBlockRows, m_numBlockColumns, (numBlocks+2), (numBlocks+2) * paddedSize<T>(blockSize) * paddedSize<T>(blockSize) );


      //
      // traverse matrix, acquire blocks and insert data
      //
      for ( int blockCol = 0; blockCol < getNumBlockRows(); ++blockCol )
      {
        for ( int blockRow = 0; blockRow <= blockCol; ++blockRow )
        {
          // block is empty
          bool haveData = false;

          // loop over rows and columns in actual block
          // slightly lazy, loop over entire column to see if there is some data
          for ( int col = (int)(blockCol * blockSize); (size_t)col < (blockCol+1)*blockSize; ++col )
            for ( int row = c[col]; row < c[col+1]; ++row )
              haveData |= (( (size_t)r[row] >= blockRow * blockSize ) && ( (size_t)r[row] < (blockRow+1)*blockSize ));

          if ( haveData )
          {
            block.setPointer( acquireBlock( blockCol, blockRow, blockSize, blockSize ) );
            block.clear();

            for ( int col = (int)(blockCol * blockSize); (size_t)col < (blockCol+1)*blockSize; ++col )
            {
              for ( int row = c[col]; row < c[col+1]; ++row )
              {
                if (( (size_t)r[row] >= blockRow * blockSize ) && ( (size_t)r[row] < (blockRow+1)*blockSize ))
                {
                  // just store lower half of block on the diagonal
                  if ( blockRow == blockCol )
                  {
                    if ( col % blockSize >= r[row] % blockSize )
                      block( col % blockSize, r[row] % blockSize ) = data[row];
                  }
                  else
                  {
                    block( col % blockSize, r[row] % blockSize ) = data[row];
                  }
                }
              }
            }
          } // haveData
        }
      } // blockRow loop
    }
  }



  // fill a sparse matrix from column compressed matrix and create a matrix with variable block size
  template< typename T >
  SparseMatrix<T>::SparseMatrix( size_t* blockSizeVector, size_t numBlockRows, size_t i, size_t j, size_t /*nnz*/, int* r, int* c, double* data )
  {
    clearStructures();

    DataBlock<T> block;

    int colLimit = 0;
    int colStartPos = 0;

    int rowLimit = 0;
    int rowStartPos = 0;

    int memory = 0;

    if ( i != j || numBlockRows > i || numBlockRows == 0 )
    {
      std::cerr << "Could not fill sparse matrix with column compressed data" << std::endl
                << "Dimension: " << i << "x" << j << ", NumBlockRows: " << numBlockRows << std::endl;
       throw;
    }
    else
    {
      int numBlocks = 0;

      m_numBlockRows = numBlockRows;
      m_numBlockColumns = numBlockRows;

      //
      // traverse matrix, count number of blocks
      //
      for ( size_t blockCol = 0; blockCol < getNumBlockRows(); ++blockCol )
      {
        colLimit = (int)(colStartPos + blockSizeVector[blockCol]);

        rowStartPos = 0;

        for ( size_t blockRow = 0; blockRow <= blockCol; ++blockRow )
        {
          rowLimit = (int)(rowStartPos + blockSizeVector[blockRow]);

          // block is empty
          bool haveData = false;

          // loop over rows and columns in actual block
          // slightly lazy, loop over entire column to see if there is some data
          for ( int col = colStartPos; col < colLimit; ++col )
            for ( int row = c[col]; row < c[col+1]; ++row )
              haveData |= (( r[row] >= rowStartPos ) && ( r[row] < rowLimit ));

          if ( haveData )
          {
            ++numBlocks;
            memory += (int)( paddedSize<T>( blockSizeVector[blockCol] ) * paddedSize<T>( blockSizeVector[blockRow] ) );
          }

          rowStartPos = rowLimit;
        }

        colStartPos = colLimit;
      }

      //
      // setup matrix, allocate memory
      //
      setSize(m_numBlockRows, m_numBlockColumns, (numBlocks+2), memory+1024 );


      colStartPos = 0;

      //
      // traverse matrix, acquire blocks and insert data
      //
      for ( size_t blockCol = 0; blockCol < getNumBlockRows(); ++blockCol )
      {
        colLimit = (int)(colStartPos + blockSizeVector[blockCol]);

        rowStartPos = 0;

        for ( size_t blockRow = 0; blockRow <= blockCol; ++blockRow )
        {
          // block is empty
          bool haveData = false;

          rowLimit = (int)(rowStartPos + blockSizeVector[blockRow]);

          // loop over rows and columns in actual block
          // slightly lazy, loop over entire column to see if there is some data
          for ( int col = colStartPos; col < colLimit; ++col )
            for ( int row = c[col]; row < c[col+1]; ++row )
              haveData |= (( r[row] >= rowStartPos ) && ( r[row] < rowLimit ));

          if ( haveData )
          {
            block.set( acquireBlock( blockCol, blockRow, blockSizeVector[blockCol], blockSizeVector[blockRow] ),
                blockSizeVector[blockCol], blockSizeVector[blockRow], paddedSize<T>( blockSizeVector[blockRow] ) );


            for ( int col = colStartPos; col < colLimit; ++col )
            {
              for ( int row = c[col]; row < c[col+1]; ++row )
              {
                if (( r[row] >= rowStartPos ) && ( r[row] < rowLimit ))
                {
                  int localCol = col - colStartPos;
                  int localRow = r[row] - rowStartPos;

                  // just store lower half of block on the diagonal
                  if ( blockRow == blockCol )
                  {
                    if ( localCol >= localRow )
                      block( localCol, localRow ) = data[row];
                  }
                  else
                  {
                    block( localCol, localRow ) = data[row];
                  }
                }
              }
            }
          } // haveData
          rowStartPos = rowLimit;

        } // blockRow loop

        colStartPos = colLimit;
      } // blockCol loop
    }
  }




  template< typename T >
  SparseMatrix<T>::SparseMatrix( const DenseMatrix<T>& d, size_t blockSize )
  {
    clearStructures();

    DataBlock<T> block(0, blockSize, blockSize);

    m_numBlockRows = d.getNumRows() / blockSize;
    m_numBlockColumns = m_numBlockRows;

    int nnzBlocks = (m_numBlockRows+1)*m_numBlockRows / 2;
    ++nnzBlocks;

    setSize( m_numBlockRows, m_numBlockColumns, nnzBlocks, nnzBlocks* paddedSize<T>(blockSize) * paddedSize<T>(blockSize) );

    for ( size_t blkRow = 0; blkRow < m_numBlockRows; ++blkRow )
    {
      for ( size_t blkCol = 0; blkCol <= blkRow; ++blkCol )
      {
        bool haveElement = false;

        for (size_t i=0; i<blockSize; ++i)
          for (size_t j=0; j<blockSize; ++j)
            haveElement |= !isZero( d( blkRow*blockSize + i, blkCol*blockSize + j ) );

        if ( haveElement )
        {
          block.setPointer( acquireBlock( blkRow, blkCol, blockSize, blockSize ) );
          for (size_t i=0; i<blockSize; ++i)
            for (size_t j=0; j<blockSize; ++j)
              block(i,j) =  d( blkRow*blockSize + i, blkCol*blockSize + j );
        }
      }
    }
  }


  template< typename T >
  SparseMatrix<T>::~SparseMatrix()
  {
    releaseMemory();
  }


  template< typename T >
  void SparseMatrix<T>::releaseMemory()
  {
    // free data
    if ( m_realData )
      delete[] m_realData;

    m_realData = 0;
    m_data = 0;

    if ( m_columnIndices )
      delete[] m_columnIndices;
    m_columnIndices = 0;

    if ( m_rowPointers )
      delete[] m_rowPointers;
    m_rowPointers = 0;

    if ( m_blockPointers )
      delete[] m_blockPointers;
    m_blockPointers = 0;

    if ( m_flags )
      delete[] m_flags;
    m_flags=0;

    if ( m_dimensions )
      delete[] m_dimensions;
    m_dimensions = 0;

    if ( m_paddedDimensions )
      delete[] m_paddedDimensions;
    m_paddedDimensions = 0;

    if ( m_dimensionSum )
      delete[] m_dimensionSum;
    m_dimensionSum = 0;

    m_allocatedDataElements = 0;
    m_allocatedBlockRows    = 0;
    m_allocatedBlocks       = 0;
  }


  template< typename T >
  void SparseMatrix<T>::clearStructures()
  {
    // no memory allocated
    m_data = 0;
    m_realData = 0;
    m_blockPointers = 0;
    m_allocatedBlockRows = 0;
    m_allocatedBlocks = 0;
    m_allocatedDataElements = 0;

    m_dimensions = 0;
    m_dimensionSum = 0;

    m_paddedDimensions = 0;

    m_columnIndices = 0;
    m_flags = 0;
    m_rowPointers = 0;

    m_numBlocks = 0;

    m_numBlockRows = m_numBlockColumns = 0;
  }


  template< typename T >
  DenseMatrix<T> SparseMatrix<T>::getAsDenseMatrix() const
  {
    DenseMatrix<T> d;

    unsigned int s = (unsigned int)getNumRows();

    d.setSize( s, s );

    int start;
    int end;

    unsigned int ld;

    for ( size_t row = 0; row < getNumBlockRows(); ++row )
    {
      start = m_rowPointers[ row ];
      end = m_rowPointers[ row+1 ];

      for ( ; start < end; ++start )
      {
        // block row, m_columnIndices[start]

        int dimI = getBlockRowDimension( row );
        int dimJ = getBlockRowDimension( m_columnIndices[ start ] );

        ld = getBlockRowPaddedDimension( m_columnIndices[ start ] );

        for (int i=0; i < dimI; ++i)
        {
          for (int j=0; j < dimJ ; ++j)
          {
            d( m_dimensionSum[row] + i, m_dimensionSum[ m_columnIndices[start] ] + j ) =
              getBlockPointer( start )[ ld * i + j ];
          }
        }
      }
    }

    return d;
  }


  template< typename T >
  AGX_FORCE_INLINE SparseRow<T> SparseMatrix<T>::row( size_t index )
  {
    SparseRow<T> r(this,index);

    return r;
  }


  template< typename T >
  AGX_FORCE_INLINE const SparseRow<T> SparseMatrix<T>::row( size_t index ) const
  {
    const SparseRow<T> r( const_cast< SparseMatrix<T>* >( this ), index );

    return r;
  }


  template< typename T >
  AGX_FORCE_INLINE T* SparseMatrix<T>::getBlockPointer( const size_t& blockIndex )
  {
    return m_blockPointers[blockIndex];
  }


  template< typename T >
  AGX_FORCE_INLINE const T* SparseMatrix<T>::getBlockPointer( const size_t& blockIndex ) const
  {
    return m_blockPointers[blockIndex];
  }


  template< typename T >
  AGX_FORCE_INLINE const unsigned int& SparseMatrix<T>::getColumnIndex( const size_t& blockIndex ) const
  {
    return m_columnIndices[blockIndex];
  }

  template< typename T >
  AGX_FORCE_INLINE const unsigned int& SparseMatrix<T>::getRowPointer( size_t r ) const
  {
    return m_rowPointers[r];
  }


  template< typename T >
  AGX_FORCE_INLINE const unsigned int* SparseMatrix<T>::getPointerVector() const
  {
    return m_rowPointers;
  }

  template< typename T >
  AGX_FORCE_INLINE const unsigned int* SparseMatrix<T>::getIndexVector() const
  {
    return m_columnIndices;
  }


  template< typename T >
  AGX_FORCE_INLINE size_t SparseMatrix<T>::getNumBlocksInRow( size_t r ) const
  {
    return
      (
        ( m_rowPointers[ r+1 ] > m_rowPointers[ r ] ) ? // valid next row pointer?
        (m_rowPointers[ r+1 ] - m_rowPointers[ r ])  : 0
      );
  }


  template< typename T >
  AGX_FORCE_INLINE size_t SparseMatrix<T>::getNumBlocks() const
  {
    return m_numBlocks;
  }


  template< typename T >
  AGX_FORCE_INLINE const unsigned int& SparseMatrix<T>::getFlag( const size_t& blockIndex ) const
  {
    return m_flags[blockIndex];
  }


  template< typename T >
  AGX_FORCE_INLINE void SparseMatrix<T>::setFlag( const size_t& blockIndex, unsigned int flag )
  {
    m_flags[blockIndex] = flag;
  }


  template< typename T >
  void SparseMatrix<T>::clear()
  {
    if ( m_data )
      memset( m_data, 0, m_allocatedDataElements * sizeof( T ) );

    if ( m_rowPointers )
      memset( m_rowPointers, 0, (m_allocatedBlockRows+1) * sizeof(unsigned int) );

    if ( m_dimensions )
      memset( m_dimensions, 0, (m_allocatedBlockRows+1) * sizeof(unsigned int) );

    if ( m_paddedDimensions )
      memset( m_paddedDimensions, 0, (m_allocatedBlockRows+1) * sizeof(unsigned int) );

    if ( m_dimensionSum )
      memset( m_dimensionSum, 0, (m_allocatedBlockRows+1) * sizeof(unsigned int) );

    m_numBlockRows = m_numBlockColumns = 0;
    m_numBlocks = 0;


    m_columnLayout.clear();
  }



  template< typename T >
  void SparseMatrix<T>::setSize( size_t rows, size_t cols, size_t blocks, size_t elems )
  {
    m_numBlockRows = rows;
    m_numBlockColumns = cols;


    if ( elems > m_allocatedDataElements )
    {
      m_allocatedDataElements = elems;

      if ( m_realData )
        delete[] m_realData;

      m_realData = new T[ elems + ( 32/sizeof(T) - 1) ];

      // align-32
      m_data = agxSabre::alignPtr( m_realData );

      memset(m_data, 0, elems*sizeof(T));
    }

    if ( blocks > m_allocatedBlocks )
    {
      if (m_columnIndices)
        delete[] m_columnIndices;
      m_columnIndices = new unsigned int[ blocks ];
      memset(m_columnIndices, 0, blocks*sizeof(unsigned int));

      if (m_flags)
        delete[] m_flags;
      m_flags = new unsigned int[ blocks ];
      memset(m_flags, 0, blocks*sizeof(unsigned int));

      if (m_blockPointers)
        delete[] m_blockPointers;
      m_blockPointers = new T*[ blocks ];
      memset(m_blockPointers, 0, blocks*sizeof(T*));

      m_allocatedBlocks = blocks;
    }


    if ( m_allocatedBlockRows < 1 + m_numBlockRows )
    {
      m_allocatedBlockRows = 1 + m_numBlockRows;
      int numElements = (int)m_allocatedBlockRows + 1;

      if ( m_rowPointers )
        delete[] m_rowPointers;

      m_rowPointers = new unsigned int[numElements];
      memset(m_rowPointers, 0, numElements*sizeof(unsigned int));

      if (m_dimensions)
        delete[] m_dimensions;
      m_dimensions = new unsigned int[numElements];
      memset(m_dimensions, 0, numElements*sizeof(unsigned int));

      if (m_paddedDimensions)
        delete[] m_paddedDimensions;
      m_paddedDimensions = new unsigned int[numElements];
      memset(m_paddedDimensions, 0, numElements*sizeof(unsigned int));


      if (m_dimensionSum)
        delete[] m_dimensionSum;
      m_dimensionSum = new unsigned int[numElements];
      memset(m_dimensionSum, 0, numElements*sizeof(unsigned int));
    }

    if (m_blockPointers)
      m_blockPointers[0] = m_data;
  }



  template< typename T >
  T* SparseMatrix<T>::acquireBlock( size_t blockRowIndex, size_t blockColIndex, size_t numRows, size_t numCols, unsigned int flag )
  {
    // perform check so that data is inserted in correct order?
    // handle empty rows?

    if ( m_numBlocks >= m_allocatedBlocks )
    {
      std::cerr << "Not enough memory, " << m_numBlocks << " " << m_allocatedBlocks << std::endl;
      agx::abort();
    }

    T* ptr = m_blockPointers[ m_numBlocks ];
    m_columnIndices[ m_numBlocks ] = (unsigned int)blockColIndex;

    m_flags[ m_numBlocks ] = flag;
    m_flags[ m_numBlocks ] = 0;

    if ( blockRowIndex > 0 )
    {
      m_dimensionSum[ blockRowIndex ] = m_dimensions[ blockRowIndex-1 ] + m_dimensionSum[ blockRowIndex-1 ];
    }


    m_dimensions[blockRowIndex] = (unsigned int)numRows;
    m_dimensions[blockColIndex] = (unsigned int)numCols;

    // We should store sum(m_dimensions) in  m_dimensions[ m_numBlockRows ]
    m_dimensions[ m_numBlockRows ] = m_dimensionSum[ blockRowIndex ] + m_dimensions[ blockRowIndex ];

    m_paddedDimensions[blockRowIndex] = (unsigned int) paddedSize<T>( m_dimensions[ blockRowIndex ] );
    m_paddedDimensions[blockColIndex] = (unsigned int) paddedSize<T>( m_dimensions[ blockColIndex ] );

    // setup next blockpointer
    if (m_numBlocks+1 < m_allocatedBlocks)
      m_blockPointers[ m_numBlocks+1 ] = m_blockPointers[ m_numBlocks ] + m_paddedDimensions[blockRowIndex] * m_paddedDimensions[ blockColIndex ];

    if ( blockRowIndex && m_rowPointers[ blockRowIndex ] == 0 )
      m_rowPointers[ blockRowIndex ] = (unsigned int)m_numBlocks;

    m_numBlocks++;

    m_rowPointers[ blockRowIndex+1] = (unsigned int)m_numBlocks;

    return ptr;
  }



  template< typename T >
  AGX_FORCE_INLINE size_t SparseMatrix<T>::getMemoryUsage() const
  {
    size_t usage = 0;

    // m_data
    usage += m_allocatedDataElements * sizeof( T );

    // m_columnIndices, m_flags, m_blockPointers
    usage += m_allocatedBlocks * (2 * sizeof( unsigned int ) + sizeof(T) );

    // m_rowPointers, m_dimensions, m_dimensionSum
    usage += m_allocatedBlockRows * 3 * sizeof( unsigned int );

    return usage;
  }

  template< typename T >
  void SparseMatrix<T>::print() const
  {
    std::cout << "SparseMatrix, " << m_numBlockRows << "x" << m_numBlockColumns << " blocks." << std::endl;

    for (size_t i = 0; i < m_numBlockRows; ++i)
    {
      row(i).print();
      std::cout << std::endl;
    }
  }


  template< typename T >
  void SparseMatrix<T>::printStructure() const
  {
    SparseMatrix<T> *nc = const_cast< SparseMatrix<T>* >( this );

    for (size_t i = 0; i<m_numBlockRows; ++i)
    {
      nc->row(i).printStructure();
      std::cout << std::endl;
    }
  }


  template< typename T >
  AGX_FORCE_INLINE const unsigned int& SparseMatrix<T>::getBlockRowDimension( size_t blockRow ) const
  {
    return m_dimensions[ blockRow ];
  }

  template< typename T >
  AGX_FORCE_INLINE const unsigned int* SparseMatrix<T>::getBlockRowDimensions() const
  {
    return m_dimensions;
  }


  template< typename T >
  AGX_FORCE_INLINE unsigned int SparseMatrix<T>::getBlockRowPaddedDimension( size_t blockRow ) const
  {
    return m_paddedDimensions[ blockRow ];
  }

  template< typename T >
  AGX_FORCE_INLINE const unsigned int* SparseMatrix<T>::getBlockRowPaddedDimensions() const
  {
    return m_paddedDimensions;
  }


  template< typename T >
  AGX_FORCE_INLINE const unsigned int& SparseMatrix<T>::getBlockRowStartRow( size_t blockRow ) const
  {
    return m_dimensionSum[ blockRow ];
  }


  template< typename T >
  size_t SparseMatrix<T>::getMaxBlockRowDimension() const
  {
    unsigned int m = 0;
    if ( m_dimensions )
    {
      m = m_dimensions[0];
      for (size_t i = 1; i < m_numBlockRows; ++i)
        m = std::max(m,m_dimensions[i]);
    }
    return m;
  }

  template< typename T >
  AGX_FORCE_INLINE size_t SparseMatrix<T>::getNumBlockRows() const
  {
    return m_numBlockRows;
  }


  template< typename T >
  size_t SparseMatrix<T>::getNumRows() const
  {

    return ( m_numBlockRows ? m_dimensions[m_numBlockRows] : 0 );
  }


  template< typename T >
  AGX_FORCE_INLINE size_t SparseMatrix<T>::getNumBlockColumns() const
  {
    return m_numBlockColumns;
  }


  template< typename T >
  AGX_FORCE_INLINE bool SparseMatrix<T>::haveAdditionalColumnLayout() const
  {
    return m_columnLayout.blkData.size() > 0;
  }


  template< typename T >
  AGX_FORCE_INLINE const MetaLayout_t& SparseMatrix<T>::getColumnLayout() const
  {
    return m_columnLayout;
  }





  /////////////////////////////////////////////////////////////////////////////
  //
  // SparseRow

  template< typename T >
  SparseRow<T>::SparseRow( SparseMatrix<T>* matrix, size_t row)
    : m_matrix(matrix), m_rowIndex( row )
  {
  }

  template< typename T >
  void SparseRow<T>::print() const
  {
    //size_t len = length();
    size_t startIndex = m_matrix->m_rowPointers[ m_rowIndex ];
    size_t numBlocks = getNumBlocks();
    size_t currentBlock = startIndex;

    DataBlock<T> block;

    // loop over num rows in blockrow
    for ( size_t i = 0; i < m_matrix->getBlockRowDimension(m_rowIndex); ++i )
    {
      currentBlock = startIndex;
      for ( size_t j = 0; j < m_matrix->getNumBlockColumns(); ++j )
      {
        if ( numBlocks && j == m_matrix->m_columnIndices[ currentBlock ] )
        {
          block.set( m_matrix->getBlockPointer( currentBlock ),
                     m_matrix->getBlockRowDimension(m_rowIndex),
                     m_matrix->getBlockRowDimension(j),
                     m_matrix->getBlockRowPaddedDimension(j) );

          // print i:th row of block
          for (size_t tmp=0; tmp<m_matrix->getBlockRowDimension(j); ++tmp)
            std::cout << std::setw(4) << std::setprecision(3) << block(i,tmp) << " ";

          currentBlock += 1 * ( currentBlock+1 < startIndex+numBlocks );
        }
        else
        {
          // print m_blocksize num zeros
          for (size_t tmp = 0; tmp < m_matrix->getBlockRowDimension(j); ++tmp)
            std::cout << std::setw(4) << 0 << " ";
        }
      }
      std::cout << std::endl;
    }
  }


  template< typename T >
  void SparseRow<T>::printStructure() const
  {
    size_t len = m_matrix->getNumBlockRows();
    size_t startIndex = m_matrix->m_rowPointers[ m_rowIndex ];
    size_t numBlocks = getNumBlocks();
    size_t currentBlock = startIndex;

    for (size_t j = 0; j < len; ++j )
    {

      if ( numBlocks && j == m_matrix->m_columnIndices[ currentBlock ] )
      {
        std::cout << "1 ";
        currentBlock += 1 * ( currentBlock+1 < startIndex+numBlocks );
      }
      else
      {
        std::cout << "0 ";
      }
      std::cout << "(" << m_matrix->getBlockRowDimension(m_rowIndex) << "x" << m_matrix->getBlockRowDimension(j) << ") ";
    }
  }


  template< typename T >
  size_t SparseRow<T>::length() const
  {
    return m_matrix->getNumRows();
  }

  template< typename T >
  size_t SparseRow<T>::getNumBlocks() const
  {
    return m_matrix->getNumBlocksInRow( m_rowIndex );
  }


  template< typename T >
  bool SparseRow<T>::operator()( size_t column, DataBlock<T>& block )
  {
    size_t start = m_matrix->m_rowPointers[ m_rowIndex ];
    size_t limit = m_matrix->m_rowPointers[ m_rowIndex + 1];

    while ( start < limit && column >= m_matrix->m_columnIndices[start] )
    {
      if ( column == m_matrix->m_columnIndices[ start ] )
      {
        DataBlock<T> tmp( m_matrix->m_blockPointers[start],
                          m_matrix->getBlockRowDimension( m_rowIndex ),
                          m_matrix->getBlockRowDimension( column ) );

        block = tmp;

        return true;
      }
      ++start;
    }
    return false;
  }



  template< typename T >
  T* SparseRow<T>::getBlockPointer( size_t column )
  {
    size_t start = m_matrix->m_rowPointers[ m_rowIndex ];
    size_t limit = m_matrix->m_rowPointers[ m_rowIndex + 1];

    while ( start < limit && column >= m_matrix->m_columnIndices[start] )
    {
      if ( column == m_matrix->m_columnIndices[ start ] )
      {
        return m_matrix->m_blockPointers[start];
      }
      ++start;
    }

    return 0;
  }


  template< typename T >
  const T* SparseRow<T>::getBlockPointer( size_t column ) const
  {
    size_t start = m_matrix->m_rowPointers[ m_rowIndex ];
    size_t limit = m_matrix->m_rowPointers[ m_rowIndex + 1];

    while ( start < limit && column >= m_matrix->m_columnIndices[start] )
    {
      if ( column == m_matrix->m_columnIndices[ start ] )
      {
        return m_matrix->m_blockPointers[ start ];
      }
      ++start;
    }

    return 0;

  }


  template< typename T >
  size_t SparseRow<T>::getFirstBlockColumnIndex() const
  {
    return m_matrix->m_rowPointers[ m_rowIndex ];
  }

  /////////////////////////////////////////////////////////////////////////////
  //
  // DataBlock

  template< typename T >
  DataBlock<T>::DataBlock() :
    m_data(0), m_cols(0), m_rows(0), m_lda(0)
  {
  }


  template< typename T >
  DataBlock<T>::DataBlock( T* data, size_t rows, size_t cols, size_t lda ) :
    m_data(data), m_cols(cols), m_rows(rows), m_lda( lda )
  {
    agxAssert( m_lda >= m_cols );
  }


  template< typename T >
  DataBlock<T>& DataBlock<T>::operator=( const DataBlock<T>& rhs )
  {
    m_data = rhs.m_data;
    m_cols = rhs.m_cols;
    m_rows = rhs.m_rows;
    m_lda  = rhs.m_lda;
    agxAssert( m_lda >= m_cols );
    return *this;
  }


  template< typename T >
  AGX_FORCE_INLINE T& DataBlock<T>::operator()(size_t i, size_t j)
  {
    return m_data[ i * m_lda + j ];
  }


  template< typename T >
  AGX_FORCE_INLINE const T& DataBlock<T>::operator()(size_t i, size_t j) const
  {
    return m_data[ i * m_lda + j ];
  }


  template< typename T >
  AGX_FORCE_INLINE T* DataBlock<T>::getPointer()
  {
    return m_data;
  }


  template< typename T >
  AGX_FORCE_INLINE const T* DataBlock<T>::getPointer() const
  {
    return m_data;
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::set( T* ptr, size_t rows, size_t cols, size_t lda )
  {
    m_data = ptr;
    m_rows = rows;
    m_cols = cols;
    m_lda  = lda;
    agxAssert( m_lda >= m_cols );
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::setPointer( T* ptr )
  {
    m_data = ptr;
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::setSize( size_t rows, size_t cols, size_t lda )
  {
    m_rows = rows;
    m_cols = cols;
    m_lda  = lda;
    agxAssert( m_lda >= m_cols );
  }


  template< typename T >
  AGX_FORCE_INLINE const size_t& DataBlock<T>::getNumRows() const
  {
    return m_rows;
  }


  template< typename T >
  AGX_FORCE_INLINE const size_t& DataBlock<T>::getNumColumns() const
  {
    return m_cols;
  }


  template< typename T >
  void DataBlock<T>::print() const
  {
    for (size_t i = 0; i < m_rows; ++i)
    {
      for (size_t j = 0; j < m_cols; ++j)
      {
        std::cout << std::setw(6) << std::setprecision(4) << m_data[i*m_lda + j] << " ";
      }
      std::cout << std::endl;
    }
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::copy( const DataBlock<T>& other )
  {
    memcpy(m_data, other.m_data, m_rows*m_lda * sizeof(T));
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::copy( const T* ptr )
  {
    memcpy(m_data, ptr, m_rows*m_lda * sizeof(T));
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::getDiagonal( T* ptr ) const
  {
    for (size_t i = 0; i<m_cols; ++i )
      ptr[i] = m_data[ i + m_lda * i ];
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::copyScaledColumns( const T* blockPtr, const T* scaleFactors )
  {
    for (size_t i = 0; i<m_rows; ++i )
    {
      for (size_t j = 0; j<m_cols; ++j )
      {
        size_t idx = i * m_lda + j;
        m_data[ idx ] = blockPtr[ idx ] * scaleFactors[j];
      }
      for ( size_t j = m_cols; j < m_lda; ++j )
        m_data[ i*m_lda + j ] = 0;
    }
  }


  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::copyInvScaledColumns( const T* blockPtr, const T* scaleFactors )
  {
    for (size_t j = 0; j<m_cols; ++j )
    {
      T inv = 1 / scaleFactors[j];
      for (size_t i = 0; i<m_rows; ++i )
      {
        int idx = int(i * m_lda + j);
        m_data[ idx ] = blockPtr[ idx ] * inv;
      }
    }
  }

  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::invScaleColumns( const T* scaleFactors )
  {
    for (size_t j = 0; j<m_cols; ++j )
    {
      T inv = 1 / scaleFactors[j];
      for (size_t i = 0; i<m_rows; ++i )
      {
        size_t idx = i * m_lda + j;
        m_data[ idx ] *= inv;
      }
    }

  }

  template< typename T >
  AGX_FORCE_INLINE void DataBlock<T>::clear()
  {
    memset( m_data, 0, m_rows * m_lda * sizeof(T) );
  }



} // namespace


#endif
