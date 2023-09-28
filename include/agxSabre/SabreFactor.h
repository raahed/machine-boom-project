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

#ifndef AGXSABRE_SABREFACTOR_H
#define AGXSABRE_SABREFACTOR_H

#include <agxSabre/agxSabre.h>
#include <agxSabre/SabreData.h>

namespace agxSabre
{

  /**
  Routines for computing matrix factorizations. Exposed via agxSabre::Sabre.
  */
  class AGXSABRE_EXPORT SabreFactor
  {

    public:


      /**
      Factors the sparse matrix H into L D L'.

      */
      template< typename T >
      static bool factor( SabreData<T>& sabredata );



      /**
      Factors changed parts of the matrix.

      For this method to work as expected, the sabredata must have been prepared
      by placing data in getEquationInsertionVector and/or
      getEquationRemovalVector. After that, prepareForModification must be used
      to translate indices and update internal structures.

      It is recommended that Sabre::modifyMatrix is used instead.
      */
      template< typename T >
      static bool partialFactor( SabreData<T>& sabredata );



    private:
      friend class Sabre;


      template< typename T >
      static bool factorSerial( SabreData<T>& sabredata );


      template< typename T >
      static void genericFactorSerial( SabreData<T> & sabredata,
                                       const unsigned char* flags = nullptr );

      template< typename T >
      static bool factorMultifrontal( SabreData<T>& sabredata, bool copyData = true );



      /*
      Compute memory needed for lower triangle factor given matrix H and elimination tree
      */
      template< typename T >
      static uint32_t computeMemoryUsage( const agxSabre::SparseMatrix<T>& H,
                                          const agxSabre::EliminationTree& eTree );


      /*
      Prepare the empty sparse matrix L by allocating blocks and
      copy data to it from the input matrix.
      */
      template< typename T >
      static void setupL( SabreData<T>& sabredata );


      /*
      Update the L matrix. Blocks must already exist and dirty parts
      are fully copied from H and then affected inner rows cleared.
      */
      template< typename T >
      static void updateL( SabreData<T>& sabredata,
                           const unsigned char* dirty,
                           const agxSabre::UInt32Vector& blockrowCounts,
                           const agxSabre::UInt32Vector& innerIndices );

      /*
      Clear equations which should not be part of factorization.
      This is similar to updateL, but no data is copied.
      */
      template< typename T >
      static void partialClearL( SabreData<T>& sabredata,
                                 const unsigned char* dirty,
                                 const agxSabre::UInt32Vector& blockrowCounts,
                                 const agxSabre::UInt32Vector& innerIndices );





  };




}

#endif
