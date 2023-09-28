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

#ifndef AGXSABRE_REORDERING_H
#define AGXSABRE_REORDERING_H

#include <agx/config.h>

#include <agxSabre/Permutation.h>
#include <agxSabre/EliminationTree.h>
#include <agxSabre/ScratchPad.h>

DOXYGEN_START_INTERNAL_BLOCK()
namespace agxSabre {

  class SkeletonMatrix;

  /**
  Reordering contains methods for computing matrix permutations.
  */
  namespace Reordering {


    ////
    // Recommended permutation methods:

    /**
    Compute an AMD permutation with the skeleton matrix and store it in p.
    */
    AGXSABRE_EXPORT bool computeSymamdPermutation( const SkeletonMatrix& m, Permutation& p, ScratchPad& pad );

    /**
    Zero-leaf swap.
    */
    AGXSABRE_EXPORT bool zeroLeafSwap( const SkeletonMatrix& m, SkeletonMatrix& pm, EliminationTree& eTree, size_t safePivots, Permutation& p, ScratchPad& pad );


    /**
    */
    AGXSABRE_EXPORT bool computeSymamdZeroSwap( const SkeletonMatrix& m, SkeletonMatrix& pm, EliminationTree& eTree, size_t safePivots, Permutation& p, ScratchPad& pad );



    /**
    "Simple minded degeneracy removal"
    */
    void modifyPermutation( const SkeletonMatrix& skeleton, size_t safePivots, Permutation& p );



  }


  inline bool agxSabre::Reordering::computeSymamdZeroSwap( const SkeletonMatrix& m, SkeletonMatrix& pm, EliminationTree& eTree, size_t safePivots,
      Permutation& p, ScratchPad& pad )
  {
    // #1. AMD ordering, elimination tree, postordering construction
    Reordering::computeSymamdPermutation( m, p, pad );

    // #2. zero-leaf swap in the e-tree.
    return Reordering::zeroLeafSwap( m, pm, eTree, safePivots, p, pad );
  }




}
DOXYGEN_END_INTERNAL_BLOCK()

#endif
