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
#include <agxSDK/MergeSplitThresholds.h>

#include <agx/IndexLambdaKernel.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( GeometryContactMergeSplitAlgorithm );

  class AGXPHYSICS_EXPORT GeometryContactMergeSplitAlgorithm : public MergeSplitAlgorithm
  {
    public:
      enum EContactState
      {
        VALID     = 1 << 0,
        IMPACTING = 1 << 1,
        SLIDING   = 1 << 2,
        ROLLING   = 1 << 3,
        RESTING   = 1 << 4
      };
      typedef agx::BitState<EContactState, agx::Int32> ContactState;

      struct ComputeData
      {
        ComputeData()
          : mergedState( (agx::RigidBody*)nullptr, nullptr, nullptr ) {}

        MergedState mergedState;
        ContactState contactState;
      };
      using ComputeDataContainer = agx::VectorPOD<ComputeData>;

    public:
      /**
      Finds the impacting state of a geometry contact.
      */
      static ContactState findImpactingContactState( const agxCollide::GeometryContact& geometryContact,
                                                     const agxSDK::MergedState& mergedState,
                                                     const agxSDK::MergeSplitHandler& handler );

      /**
      Finds the contact state of a geometry contact after the solver has solved the contact.
      */
      static ContactState findContactState( const agxCollide::GeometryContact& geometryContact,
                                            const agxSDK::MergedState& mergedState,
                                            const agxSDK::MergeSplitPostSolveData& postSolveData,
                                            const agxSDK::MergeSplitHandler& handler );

    public:
      /**
      Default constructor.
      */
      GeometryContactMergeSplitAlgorithm();

      /**
      \note This data is only valid during postSolve given that this algorithm has been updated.
      \return post solve compute data
      */
      const ComputeDataContainer& getComputeData() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::GeometryContactMergeSplitAlgorithm );

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~GeometryContactMergeSplitAlgorithm();

      virtual void onPreStepCompute() override;
      virtual void onPreStepActions( agxSDK::MergeSplitActionContainer& actions ) override;

      virtual void onPostSolveCompute( const agxSDK::MergeSplitPostSolveData& postSolveData ) override;
      virtual void onPostSolveActions( const agxSDK::MergeSplitPostSolveData& postSolveData, agxSDK::MergeSplitActionContainer& actions ) override;

    private:
      agx::IndexLambdaKernelRef m_kernel;
      ComputeDataContainer m_result;
  };
}
