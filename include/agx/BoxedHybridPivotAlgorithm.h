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

#ifndef AGX_BOXEDHYBRIDPIVOTALGORITHM_H
#define AGX_BOXEDHYBRIDPIVOTALGORITHM_H

#include <agx/BoxedMurtyAlgorithm.h>
#include <agx/BoxedKellerAlgorithm.h>

namespace agx
{
  class AGXPHYSICS_EXPORT BoxedHybridPivotAlgorithm : public agx::NlMixedCp::McpAlgorithm
  {
    public:
      /**
      Construct given global tolerance and the maximum iterations.
      */
      BoxedHybridPivotAlgorithm( agx::Real globalTolerance, agx::UInt maxNumIterations, bool gatherStatistics );

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~BoxedHybridPivotAlgorithm();

      /**
      Solves MCP using Keller's Principal Pivot Method.
      */
      virtual agx::NlMixedCp::McpAlgorithm::SolveReport solve( agx::NlMixedCp::McpAlgorithm::Args& args ) override;

    protected:
      agx::Real m_localTolerance;
      agx::BoxedMurtyAlgorithmRef m_blockPivot;
      agx::BoxedKellerAlgorithmRef m_keller;
  };

  typedef agx::ref_ptr< BoxedHybridPivotAlgorithm > BoxedHybridPivotAlgorithmRef;
}

#endif
