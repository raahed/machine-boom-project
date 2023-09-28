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

#ifndef AGXSABRE_TREEUPDATE_H
#define AGXSABRE_TREEUPDATE_H

/**
\file TreeUpdate.h

*/

#include <agxSabre/agxSabre.h>
#include <agxSabre/SabreUtils.h>

namespace agxSabre
{

  class EliminationTree;

  class AGXSABRE_EXPORT TreeUpdate
  {
    public:

      /**
      Remove node from parent and make sure that:
      - parents firstChild and sibling structure is still valid
      - node will become it's own parent.
      */
      static void removeChild(  agxSabre::EliminationTree* etree, uint32_t node, uint32_t parent );

      /**
      Inserts a new node under parent.
      Node is not allowed to have any siblings, then by definition, it must already have a parent.
      */
      static void insertChild(  agxSabre::EliminationTree* etree, uint32_t node, uint32_t parent );


      static void updateParent( agxSabre::EliminationTree* etree, uint32_t node, uint32_t oldParent, uint32_t newParent );


      /**
      Move away all children that node has and place them under newParent
      */
      static void moveChildren( agxSabre::EliminationTree* etree, uint32_t node, uint32_t newParent );


      static uint32_t findFirstBlock( const agxSabre::SparseMatrix<double>& L,
          const MetaLayout_t& meta, uint32_t blkCol );


      static void computeTree( agxSabre::EliminationTree* etree,
                               const agxSabre::SparseMatrix<double>& L,
                               const MetaLayout_t& meta );

     static void computeTree( agxSabre::EliminationTree* etree,
                               const agxSabre::SkeletonMatrix& skel,
                               const uint32_t* activeInfo );


      /**
      Check if the Elimination Tree etree is valid for the lower triangular matrix L.
      */
      static bool validate( agxSabre::EliminationTree* etree,
                            const agxSabre::SparseMatrix<double>& L,
                            const MetaLayout_t& meta );

  };


}

#endif


