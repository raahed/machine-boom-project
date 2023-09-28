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

#pragma once

#include <agx/Integer.h>
#include <agx/Real.h>

namespace agx
{
  /**
  Interface to a row in the compliance matrix.
  */
  class LSquareComplianceMatrixRow
  {
    public:
      /**
      Construct given row data and number of accessible columns.
      \param data - row data
      \param size - number of columns
      */
      inline LSquareComplianceMatrixRow( Real* data, UInt32 size )
        : m_data( data )
        , m_size( size )
      {
      }

      /**
      Write access to value in given column.
      \param col - row column
      \return value at given column
      */
      inline Real& operator[]( UInt32 col )
      {
        agxAssert( col < m_size );
        return m_data[ col ];
      }

      /**
      Read access to value in given column.
      \param col - row column
      \return value at given column
      */
      inline Real operator[]( UInt32 col ) const
      {
        agxAssert( col < m_size );
        return m_data[ col ];
      }

      /**
      \return number of accessible columns in this row
      */
      inline UInt32 size() const
      {
        return m_size;
      }

    private:
      Real* m_data;
      UInt32 m_size;
  };

  /**
  Interface to the lower triangle of the compliance
  matrix block in the system matrix. All values written
  to this matrix block has to be less than or equal to zero.

  The lower triangle is written meaning row 0 has 1 column,
  row 1 has 2 columns etc. to write to.
  */
  class LSquareComplianceMatrix
  {
    public:
      /**
      Construct given data, number of rows/columns and leading dimension.
      \param data - matrix data
      \param size - number of rows/columns
      \param ld - leading dimension of the data
      */
      inline LSquareComplianceMatrix( Real* data, UInt32 size, UInt32 ld )
        : m_data( data )
        , m_size( size )
        , m_ld( ld )
      {
      }

      /**
      Read/write access to given row number.
      \param row - row to access
      \return perturbation row interface
      */
      inline LSquareComplianceMatrixRow operator[]( UInt32 row ) const
      {
        agxAssert( row < m_size );
        return LSquareComplianceMatrixRow{ m_data + row * m_ld, row + 1u };
      }

      /**
      \return the number of rows and max number of columns in this matrix
      */
      inline UInt32 size() const { return m_size; }

    private:
      Real* m_data;
      UInt32 m_size;
      UInt32 m_ld;
  };
}
