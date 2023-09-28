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

#include <agxSDK/MergeSplitAlgorithm.h>
#include <agxSDK/MergedBodySolverData.h>

#include <agx/IndexLambdaKernel.h>

namespace agxSDK
{
  class AGXPHYSICS_EXPORT MergedBodyMergeSplitAlgorithm : public agxSDK::MergeSplitAlgorithm
  {
    public:
      struct PostSolveMergedBodySolverData
      {
        MergedBodySolverData solverData;
      };
      using PostSolveMergedBodySolverDataContainer = agx::Vector<PostSolveMergedBodySolverData>;

    public:
      MergedBodyMergeSplitAlgorithm();

      const MergedBodySolverData* getMergedBodySolverData( const agx::MergedBody* mergedBody ) const;
      const MergedBodySolverData* getMergedBodySolverDataGivenIndex( agx::UInt index ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::MergedBodyMergeSplitAlgorithm );

    protected:
      virtual ~MergedBodyMergeSplitAlgorithm();

      virtual void onPostSolveCompute( const agxSDK::MergeSplitPostSolveData& postSolveData ) override;

      friend class MergeSplitHandler;
      virtual void onPostSolveComputeCollectData( const agxSDK::MergeSplitPostSolveData& postSolveData );

    private:
      using MergedBodySolverDataCache = agx::HashTable<const agx::MergedBody*, const MergedBodySolverData*>;

    private:
      agx::IndexLambdaKernelRef m_kernel;
      PostSolveMergedBodySolverDataContainer m_postSolveResult;
      MergedBodySolverData::MergedBodyGraphNodeIndexContainer m_staticMergedBodiesExternalInteractions;
      MergedBodySolverDataCache m_mergedBodySolverDataCache;
  };
}
