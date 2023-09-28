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

#ifndef AGX_SPARSE_MATRIX_H
#define AGX_SPARSE_MATRIX_H

#include <agx/Referenced.h>
#include <agx/HashTable.h>

#include <agx/agx_valarray_types.h>
#include <agx/MatrixTypes.h>

namespace agx
{

  AGX_DECLARE_POINTER_TYPES(SparseMatrixRep);

  /**
  SparseMatrixRep
  */
  class AGXPHYSICS_EXPORT SparseMatrixRep : public Referenced
  {
    public:
      virtual agx::Real* getBlock(int i, int j, bool& writeTransposed, unsigned int& lda) = 0;

      virtual void reset() = 0;

      virtual void factor() = 0;
      virtual void solve( agx::Real* ) = 0;

      virtual size_t unitSolve( agx::Real*, int = -1 );

      virtual void multiply( const agx::Real* x, agx::Real* r ) = 0;

      virtual void multiplyAdd( const agx::Real* x, agx::Real* r ) = 0;

      virtual void multiplyAdd( const agx::Real& alpha, const agx::Real* x, const agx::Real& beta, agx::Real* r ) = 0;

      virtual void setLogicalStructure( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions, const int* flags, unsigned int numThreads );

      virtual agx::Bool getRow( agx::UInt rowNumber, const agx::RealValarray& signs, agx::RealValarray& retRow ) const = 0;
      virtual agx::Bool getColumn( agx::UInt colNumber, agx::RealValarray& retCol, agx::Bool& signFlip ) const = 0;

      virtual agx::UInt getBlockPermutedStartRowIndex( agx::UInt blockIndex ) const = 0;
      virtual agx::UInt permuteRowIndex( agx::UInt rowIndex ) const;
      virtual agx::UInt invPermuteRowIndex( agx::UInt permutedRowIndex ) const;

      virtual unsigned int getNumRows() const = 0;
      virtual unsigned int getNumColumns() const = 0;

      virtual void print() const {}

      virtual bool saveH5(void* group, const agx::String& name ) const;
      virtual bool loadH5(void* group, const agx::String& name );

      virtual void setNumBodies(size_t /*num*/) {}

      virtual agx::Real getFactorTime() const { return -1; }
      virtual agx::Real getSolveTime() const { return -1; }

    protected:
      virtual ~SparseMatrixRep() {}
 };


  /**
  agx::SparseMatrix holds a matrix representation which contains the data.

  Each solver wrapper (if it can handle the matrix format defined by MatrixType),
  uses this data and passes it in the correct format to the solver.
  */
  class AGXPHYSICS_EXPORT SparseMatrix
  {
    public:

      /**
      Flags that can be used to tag blocks so that the matrix representation
      can optimize the calculations.
      */
      enum BlockFlags {
        DENSE     = 0,
        DIAGONAL  = 1,
        LOWER_TRIANGULAR = 2,
        UPPER_TRIANGULAR = 3
      };


      /**
      Constructor.
      \param type Matrix type
      \param rep Pointer to SparseMatrixRepresentation. The SparseMatrix takes ownership of the pointer.
      */
      SparseMatrix( MatrixType type = MatrixType(), SparseMatrixRep* rep = 0 );

      /**
      Destructor
      */
      ~SparseMatrix();

      /**
      Sets the logical structure of the matrix.
      The method takes a matrix in packed row format.
      \param n Number of (block) rows
      \param pointers Row pointers
      \param indices Column indices
      \param dimensions Number of rows for each block row. If not blocked, each dimension[i] will be 1.
      \param flags Pointer to flag for each block, if 0, all blocks are assumed to be dense.
      \param numThreads Hint about maximum number of threads the factorizer can use.
      */
      void setLogicalStructure( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions, const int* flags = nullptr, unsigned int numThreads = 1 );

      /**
      Extract a row from the matrix.
      \param rowNumber - row number to extract
      \param signs - sign array -1 for constraints, 1 for bodies
      \param[out] retRow - result
      \return true if successful
      */
      agx::Bool getRow( agx::UInt rowNumber, const agx::RealValarray& signs, agx::RealValarray& retRow ) const;

      /**
      Extract a column from the matrix.
      \param colNumber - column number to extract
      \param[out] retCol - result
      \return true if successful
      */
      agx::Bool getColumn( agx::UInt colNumber, agx::RealValarray& retCol ) const;


      /**
      Extract a column from the matrix.
      \param colNumber - column number to extract
      \param[out] retCol - result
      \param[out] signFlip - depending on \p colNumber the sign has to be flipped when this column is transformed to a row
      \return true if successful
      */
      agx::Bool getColumn( agx::UInt colNumber, agx::RealValarray& retCol, agx::Bool& signFlip ) const;

      /**
      Fetches the location for a block of the matrix. The block can
      be 1x1 in size for non block matrix.
      \param i Block row index.
      \param j Block column index.
      \param leadingDimension Leading dimension when writing the block data. Row major/column major
                              depends on MatrixRep used.
      */
      agx::Real* getBlock(int i, int j, bool& writeTransposed, unsigned int& leadingDimension);

      /**
      Clears the matrix.
      */
      void reset();

      /**
      Factors the matrix.
      */
      void factor();

      /**
      Solve, the right hand side is overwritten with the result.
      */
      void solve( agx::Real* );

      /**
      Solve, the right hand side is overwritten with the result.
      The right hand side is contains one 1, the rest is 0.
      */
      size_t unitSolve( agx::Real*, int = -1 );

      /**
      \return Number of rows in the matrix.
      */
      unsigned int getNumRows() const;

      /**
      \return Number of columns in the matrix.
      */
      unsigned int getNumColumns() const;

      /**
      Calculates r = A x
      */
      void multiply( const agx::Real* x, agx::Real* r );

      /**
      Calculates r = Ax + r
      */
      void multiplyAdd( const agx::Real* x, agx::Real* r );

      /**
      Calculates r = alpha A x + beta r
      */
      void multiplyAdd( const agx::Real& alpha, const agx::Real* x, const agx::Real& beta, agx::Real* r );

      /**
      \param blockIndex - non-permuted block index
      \return permuted start row
      */
      agx::UInt getBlockPermutedStartRowIndex( agx::UInt blockIndex ) const;

      /**
      \param rowIndex - non-permuted rowIndex
      \return permuted row index
      */
      agx::UInt permuteRowIndex( agx::UInt rowIndex ) const;

      /**
      \param permutedRowIndex - permuted row index
      \return non-permuted start row index
      */
      agx::UInt invPermuteRowIndex( agx::UInt permutedRowIndex ) const;

      /**
      \return Type information about the matrix
      */
      MatrixType& getType();


      /**
      */
      void print() const;

      /**
      Stores the matrix in HDF5 format.
      \param group Pointer to H5::Group in which matrix should be stored
      \param name Name to use when storing matrix
      \return true if successful
      */
      bool saveH5(void* group, const agx::String& name ) const;

      /**
      Loads the matrix
      \param group Pointer to H5::Group in which matrix should be stored
      \param name Name to use when loading matrix
      \return true if successful
      */
      bool loadH5(void* group, const agx::String& name );

      /**
      Sets the number of bodies, can be used by the representation to
      help making the permutation.
      */
      void setNumBodies( int num );

      /// \todo The abstraction layer has to be redesigned so the LCP solver
      //        can perform the necessary operations in a reasonable way.
      SparseMatrixRep *getRep();
      const SparseMatrixRep* getRep() const;


      /**
      Replace the Sparse Matrix Representation.
      The SparseMatrix takes ownership of the SparseMatrixRep

      This method is only needed in some very special use cases.
      */
      void setRep(SparseMatrixRep *rep);

      agx::Real getFactorTime() const;
      agx::Real getSolveTime() const;

    private:
      SparseMatrixRepRef m_rep;
      MatrixType m_type;
  };

  /**
  Class that stores rows and columns of the matrix.
  */
  class ColumnStorage
  {
    public:
      ColumnStorage( const SparseMatrix& matrix, const RealValarray& signs )
        : m_matrix( matrix ), m_signs( signs ) {}

      const RealValarray& getOrCreateColumn( UInt col )
      {
        return getOrCreateColRow( col )->column;
      }

      const RealValarray& getOrCreateRow( UInt row )
      {
        return getOrCreateColRow( row )->asRow( m_signs );
      }

    private:
      struct ColRow : public Referenced
      {
        RealValarray column;
        RealValarray row;
        Bool signFlip;

        const RealValarray& asRow( const RealValarray& signs )
        {
          if ( column.size() == 0 )
            throw;

          if ( row.size() == column.size() )
            return row;
          else
            row.resize( column.size() );

          row = column;
          Real sign = signFlip ? Real( -1 ) : Real( 1 );
          for ( UInt i = 0, numRows = row.size(); i < numRows; ++i )
            row[ i ] *= sign * signs[ i ];
          return row;
        }
      };

      typedef ref_ptr< ColRow > ColRowRef;

    private:
      typedef HashTable< UInt, ColRowRef > ColRowContainer;

    private:
      ColumnStorage& operator = ( const ColumnStorage& ) { return *this; }

      ColRowRef getOrCreateColRow( UInt col )
      {
        ColRowContainer::iterator it = m_colRowStorage.find( col );
        if ( it == m_colRowStorage.end() ) {
          it = m_colRowStorage.insert( col, new ColRow() );
          m_matrix.getColumn( col, it->second->column, it->second->signFlip );
        }

        return it->second;
      }

    private:
      const SparseMatrix& m_matrix;
      const RealValarray& m_signs;
      ColRowContainer m_colRowStorage;
  };

  /////////////////////////////////////////////////////////////////////////////
  //
  // SparseMatrix

  AGX_FORCE_INLINE MatrixType& SparseMatrix::getType()
  {
    return m_type;
  }


  AGX_FORCE_INLINE void SparseMatrix::setLogicalStructure(
    size_t n, const unsigned int* pointers, const unsigned int* indices,
    const unsigned int* dimensions, const int* flags, unsigned int numThreads )
  {
    m_rep->setLogicalStructure(n,pointers, indices, dimensions, flags, numThreads);
  }

  AGX_FORCE_INLINE agx::Real* SparseMatrix::getBlock(int i, int j, bool& writeTransposed, unsigned int& leadingDimension)
  {
    return m_rep->getBlock(i, j, writeTransposed, leadingDimension);
  }

  AGX_FORCE_INLINE void SparseMatrix::reset()
  {
    m_rep->reset();
  }

  AGX_FORCE_INLINE void SparseMatrix::factor()
  {
    m_rep->factor();
  }

  AGX_FORCE_INLINE void SparseMatrix::solve( agx::Real* rhs)
  {
    m_rep->solve( rhs );
  }

  AGX_FORCE_INLINE size_t SparseMatrix::unitSolve( agx::Real* rhs, int startRow /*= -1*/ )
  {
    return m_rep->unitSolve( rhs, startRow );
  }

  AGX_FORCE_INLINE void SparseMatrix::multiply( const agx::Real* x, agx::Real* r )
  {
    m_rep->multiply(x,r);
  }

  AGX_FORCE_INLINE void SparseMatrix::multiplyAdd( const agx::Real* x, agx::Real* r )
  {
    m_rep->multiplyAdd(x, r);
  }

  AGX_FORCE_INLINE void SparseMatrix::multiplyAdd( const agx::Real& alpha, const agx::Real* x, const agx::Real& beta, agx::Real* r )
  {
    m_rep->multiplyAdd(alpha, x, beta, r );
  }

  inline UInt SparseMatrix::getBlockPermutedStartRowIndex( UInt blockIndex ) const
  {
    return m_rep->getBlockPermutedStartRowIndex( blockIndex );
  }

  inline UInt SparseMatrix::permuteRowIndex( UInt rowIndex ) const
  {
    return m_rep->permuteRowIndex( rowIndex );
  }

  inline UInt SparseMatrix::invPermuteRowIndex( UInt permutedRowIndex ) const
  {
    return m_rep->invPermuteRowIndex( permutedRowIndex );
  }

  AGX_FORCE_INLINE unsigned int SparseMatrix::getNumRows() const
  {
    return m_rep->getNumRows();
  }

  AGX_FORCE_INLINE unsigned int SparseMatrix::getNumColumns() const
  {
    return m_rep->getNumColumns();
  }

  inline void SparseMatrix::print() const
  {
    return m_rep->print();
  }

  inline bool SparseMatrix::saveH5(void* group, const agx::String& name ) const
  {
    return m_rep->saveH5( group, name );
  }

  inline bool SparseMatrix::loadH5(void* group, const agx::String& name )
  {
    return m_rep->loadH5( group, name );
  }

  AGX_FORCE_INLINE void SparseMatrix::setNumBodies( int num )
  {
    m_rep->setNumBodies( num );
  }
}
#endif
