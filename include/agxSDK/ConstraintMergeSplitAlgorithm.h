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

#include <agxUtil/agxUtil.h>

#include <agx/IndexLambdaKernel.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( ConstraintMergeSplitAlgorithm );

  class AGXPHYSICS_EXPORT ConstraintMergeSplitAlgorithm : public agxSDK::MergeSplitAlgorithm
  {
    public:
      enum EConstraintState
      {
        REL_VELOCITIES_BELOW_THRESHOLD        = 1 << 0,
        DRIVER_ACTIVE                         = 1 << 1,
        SPLIT_FIRST_DUE_TO_CONSTRAINT_FORCES  = 1 << 2,
        SPLIT_SECOND_DUE_TO_CONSTRAINT_FORCES = 1 << 3,
        ACTIVE_TOGGLED                        = 1 << 4,
        DIFF_ABOVE_THRESHOLD                  = 1 << 5
      };
      typedef agx::BitState<EConstraintState, agx::Int32> ConstraintState;

      struct ComputeData
      {
        ComputeData()
          : mergedState( ConstraintMergedState::invalid() )
        {
        }

        ConstraintMergedState mergedState;
        ConstraintState constraintState;
      };
      using ComputeDataContainer = agx::VectorPOD<ComputeData>;

      struct PostSolveComputeData : public ComputeData
      {
        PostSolveComputeData()
          : ComputeData() {}

        agx::Vec3Vector forces;
      };
      using PostSolveComputeDataContainer = agx::Vector<PostSolveComputeData>;

      struct PostSolveComputeExData
      {
        agx::RigidBodyPtrVector bodiesToSplit;
      };
      using PostSolveComputeExDataContainer = agx::Vector<PostSolveComputeExData>;

    public:
      /**
      Default constructor.
      */
      ConstraintMergeSplitAlgorithm();

      /**
      \note This data is only valid during postSolve given that this algorithm has been updated.
      \return post solve compute data
      */
      const PostSolveComputeDataContainer& getPostSolveComputeData() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::ConstraintMergeSplitAlgorithm );

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~ConstraintMergeSplitAlgorithm();

      virtual void onPreStepCompute() override;
      virtual void onPreStepActions( agxSDK::MergeSplitActionContainer& actions ) override;

      virtual void onPostSolveCompute( const agxSDK::MergeSplitPostSolveData& postSolveData ) override;
      virtual void onPostSolveComputeEx( const agxSDK::MergeSplitPostSolveData& postSolveData,
                                         const agxSDK::MergedBodyMergeSplitAlgorithm& mergedBodyAlgorithm ) override;
      virtual void onPostSolveActions( const agxSDK::MergeSplitPostSolveData& postSolveData, agxSDK::MergeSplitActionContainer& actions ) override;

    private:
      void propagateSplit( const agx::HighLevelConstraintImplementation& implementation,
                           const agxSDK::ConstraintMergedState& mergedState,
                           agxSDK::MergeSplitActionContainer& actions );

    private:
      agx::IndexLambdaKernelRef m_kernel;
      ComputeDataContainer m_preStepResult;
      PostSolveComputeDataContainer m_postSolveResult;
      PostSolveComputeExDataContainer m_postSolveExResult;
  };
}
