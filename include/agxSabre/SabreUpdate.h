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

#ifndef AGXSABRE_SABREUPDATE_H
#define AGXSABRE_SABREUPDATE_H

/**
\file SabreUpdate.h

*/

#include <agxSabre/agxSabre.h>
#include <agxSabre/SabreData.h>
#include <agxSabre/SabreUtils.h>

namespace agxSabre
{

  class AGXSABRE_EXPORT SabreUpdate
  {

    public:


      /**
      Update the flag in sabredata which indicates that rank change methods have been used.

      Preferred way is to use addAndDeleteEquations which takes care of this step automatically.
      */
      static void markUpDownUsage( agxSabre::SabreData<double>& sabredata );


      /**
      Method to change which equations which should be part of matrix factorization.
      This method will use deleteRows and addRows as well as make sure MetaLayouts exists
      where needed.

      rankData->removeEqs and rankData->insertEqs should contain the equations that
      should be changed.
      */
      static void addAndDeleteEquations( SabreData<double>& sabredata );


      /**
      Read equation from H and store in C in column cIndex.
      \return First block idx
      */
      static uint32_t readEquation( const agxSabre::SparseMatrix<double>& H,
                                    agxSabre::DenseMatrix<double>& C,
                                    const uint32_t cIndex,
                                    const agxSabre::UInt32Pair equation,
                                    const MetaLayout_t& meta );









      /**
      Reinsert equations by modifying L and D. Data is read from H.
      */
      static bool addRows(     agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>&  D,
                               const agxSabre::SparseMatrix<double>& H,
                               const agxSabre::UInt32PairVector& equations,
                               const MetaLayout_t& metaL,
                               const MetaLayout_t& metaH,
                               agxSabre::EliminationTree* etree,
                               bool haveMatchingTree );


      /**
      Batch-reinsertion of equations by modifying L and D.
      */
      static bool addKRows(    agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>&  D,
                               const agxSabre::SparseMatrix<double>& H,
                               const agxSabre::UInt32PairVector& equations,
                               const MetaLayout_t& metaL,
                               const MetaLayout_t& metaH,
                               agxSabre::EliminationTree* etree );





      /**
      Performs multiple rank-1 updates and tree updates.
      */
      static bool deleteRows(  agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>& D,
                               const agxSabre::UInt32PairVector& equations,
                               const MetaLayout_t& meta,
                               agxSabre::EliminationTree* etree,
                               bool haveMatchingTree );

      /**
      Performs one rank-k update and tree update.
      */
      static bool deleteKRows( agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>& D,
                               const agxSabre::UInt32PairVector& equations,
                               const MetaLayout_t& meta,
                               agxSabre::EliminationTree* etree = nullptr );

      /**
      Rank-1 update.

      L n-n
      D n

      w k-n

      wIndex which ROW of w to use

      */
      static bool rank1Update( agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>& D,
                               agxSabre::DenseMatrix<double>& w,
                               double alfa,
                               const MetaLayout_t& meta,
                               agxSabre::EliminationTree* etree,
                               uint32_t treeNode,
                               uint32_t wIndex,
                               bool haveMatchingTree
                               );




      /**
      Rank-k update.

      L     n-n
      D     n

      w     k-n
      alfa  n
      */
      static bool rankKUpdate( agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>& D,
                               agxSabre::DenseMatrix<double>& w,
                               agxSabre::DenseVector<double>& alfa,
                               const MetaLayout_t& meta,
                               agxSabre::EliminationTree* etree,
                               std::vector<int>& queueA,
                               std::vector<int>& queueB
                               );






   private:


      static void insertData(  agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>&  data,
                               agxSabre::UInt32Pair equation,
                               const MetaLayout_t& meta );




      /**
      Reinsert equations by modifying L and D. Data is read from H.
      */
      static bool addRowsV2(   agxSabre::SparseMatrix<double>& L,
                               agxSabre::DenseVector<double>&  D,
                               const agxSabre::SparseMatrix<double>& H,
                               const agxSabre::UInt32PairVector& equations,
                               const MetaLayout_t& metaL,
                               const MetaLayout_t& metaH,
                               agxSabre::EliminationTree* etree,
                               uint32_t* activeEqInfo = nullptr,
                               agxSabre::SkeletonMatrix* skel = nullptr);

      /**
      Performs multiple rank-1 updates and tree updates.
      */
      static bool deleteRowsV2( agxSabre::SparseMatrix<double>& L,
                                agxSabre::DenseVector<double>& D,
                                const agxSabre::UInt32PairVector& equations,
                                const MetaLayout_t& meta,
                                agxSabre::EliminationTree* etree = nullptr,
                                uint32_t* activeEqInfo = nullptr,
                                agxSabre::SkeletonMatrix* skel = nullptr);


      static bool rank1UpdateV2( agxSabre::SparseMatrix<double>& L,
                                 agxSabre::DenseVector<double>& D,
                                 agxSabre::DenseMatrix<double>& w,
                                 double alfa,
                                 const MetaLayout_t& meta,
                                 agxSabre::EliminationTree* etree = nullptr,
                                 uint32_t treeNode = 0,
                                 uint32_t wIndex = 0 );



      static uint32_t readEquation( const agxSabre::SparseMatrix<double>& H,
                                    agxSabre::DenseVector<double>& C,
                                    const agxSabre::UInt32Pair equation,
                                    const MetaLayout_t& meta,
                                    agxSabre::UInt32Set& pattern );


      /**
      This version of funnyMult loops over the rows equation+1:end in L
      and multiplies column 1:equation-1 with work 1:equation-1 and stores the result
      int work equation+1:end.
      */
      static void funnyMult(   const agxSabre::SparseMatrix<double>& L,
                               const agxSabre::DenseVector<double>& D,
                               agxSabre::DenseVector<double>& work,
                               agxSabre::UInt32Pair equation );

      /**
      This version of funnyMult uses the pattern information which
      says where in work 1:equation-1 there are non-zero elements.

      The multiplication is then performed as work(eq+1:end) += work(j) * L(eq+1:end, j );
      */
      static void funnyMult(   const agxSabre::SparseMatrix<double>& L,
                               const agxSabre::DenseVector<double>& D,
                               agxSabre::DenseVector<double>& work,
                               agxSabre::UInt32Pair equation,
                               const MetaLayout_t& meta,
                               const agxSabre::UInt32Set& pattern );





      static bool rank1UpdateRM( agxSabre::SparseMatrix<double>& L,
                                 agxSabre::DenseVector<double>& D,
                                 agxSabre::DenseMatrix<double>& w,
                                 double alfa,
                                 const MetaLayout_t& meta,
                                 agxSabre::EliminationTree* etree,
                                 uint32_t treeNode,
                                 uint32_t wIndex,
                                 bool haveMatchingTree);



  };

}

#endif


