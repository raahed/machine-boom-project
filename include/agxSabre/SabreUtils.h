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

#ifndef AGXSABRE_SABREUTILS_H
#define AGXSABRE_SABREUTILS_H

/**
\file SabreUtils.h

*/

#include <agx/agx.h>
#include <agx/macros.h>

#include <agxSabre/agxSabre.h>
#include <agxSabre/SabreData.h>
#include <agxSabre/MetaLayout.h>

namespace agxSabre
{



  /**
  Find matrix blockrow index pair from ordinary index
  */
  template< typename T >
  inline agxSabre::UInt32Pair findBlockRow( const agxSabre::SparseMatrix<T>& M, uint32_t row );


  /**
  Translate from row index in unpermuted matrix to pair with blockrow and innerrow.
  */
  template< typename T >
  void translateIndices( const agxSabre::SabreData<T>& sd, const agxSabre::UInt32Vector& globalRows, agxSabre::UInt32PairVector& translated );

  template< typename T >
  agxSabre::UInt32PairVector translateIndices( const agxSabre::SabreData<T>& sd, const agxSabre::UInt32Vector& globalRows );




}





template< typename T >
inline agxSabre::UInt32Pair findBlockRow( const agxSabre::SparseMatrix<T>& M, uint32_t row )
{
  agxSabre::UInt32Pair result;

  uint32_t bIdx = 0;

  while ( M.getBlockRowStartRow( bIdx ) + M.getBlockRowDimension( bIdx ) <= row )
    ++bIdx;

  result.first  = bIdx;
  result.second = row - M.getBlockRowStartRow( bIdx);

  return result;
}




template< typename T >
void agxSabre::translateIndices( const agxSabre::SabreData<T>& sd, const agxSabre::UInt32Vector& globalRows, agxSabre::UInt32PairVector& translated )
{
  translated.clear();
  translated.reserve( globalRows.size() );

  const agxSabre::SparseMatrix<T>& H = sd.getMatrix();
  uint32_t numBlockRows = (uint32_t) H.getNumBlockRows();

  for ( uint32_t g = 0; g < globalRows.size(); ++g )
  {
    uint32_t blockRow = 0;
    uint32_t innerRow = 0;

    uint32_t target = globalRows[g];

    for ( uint32_t i = 0; i < numBlockRows; ++i )
    {
      uint32_t start = H.getBlockRowStartRow( i );
      uint32_t end   = start + H.getBlockRowDimension( i );

      if ( target >= start && target < end )
      {
        blockRow = i;
        innerRow = target - start;
        translated.push_back( agxSabre::UInt32Pair(blockRow, innerRow) );
        break;
      }
    }
  }
}



template< typename T >
agxSabre::UInt32PairVector agxSabre::translateIndices( const agxSabre::SabreData<T>& sd,
                                                    const agxSabre::UInt32Vector& globalRows )
{
  agxSabre::UInt32PairVector ret;
  translateIndices( sd, globalRows, ret );
  return ret;
}




#endif
