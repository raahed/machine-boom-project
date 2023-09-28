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

#include <agxSDK/MergeSplitPostSolveData.h>
#include <agxSDK/MergeSplitAction.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( MergeSplitAlgorithm );
  class MergedBodyMergeSplitAlgorithm;

  /**
  Stateless base class for merge and split of interactions. Stateless in
  the sense that it's not defined to store any kind of state in an implementation
  of this interface.
  */
  class AGXPHYSICS_EXPORT MergeSplitAlgorithm : public agx::Referenced, public agxStream::Serializable
  {
    public:
      enum Callback
      {
        NONE             = 0,
        CONTACTS         = 1 << 0,
        CONSTRAINTS      = 1 << 1,
        WIRES            = 1 << 2,
        POST_SOLVE       = 1 << 3,
        GRANULARS        = 1 << 4,
        END_OF_CALLBACKS = 1 << 5
      };

    public:
      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxSDK::MergeSplitAlgorithm );

    protected:
      /**
      Default constructor.
      */
      MergeSplitAlgorithm();

      /**
      \return the merge split handler this algorithm belongs to
      */
      agxSDK::MergeSplitHandler* getMergeSplitHandler() const;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~MergeSplitAlgorithm();

      /**
      Compute callback in preStep. Compute the data used to decide merge and/or split actions
      to push in onPreStepActions.
      */
      virtual void onPreStepCompute();

      /**
      Push merge and/or split actions in simulation preStep.
      \param[out] actions - resulting actions
      */
      virtual void onPreStepActions( agxSDK::MergeSplitActionContainer& actions );

      /**
      Compute callback in postSolve. Compute the data used to decide merge and/or split actions
      to push in onPostStepActions.
      \param postSolveData - the post solve data
      */
      virtual void onPostSolveCompute( const agxSDK::MergeSplitPostSolveData& postSolveData );

      /**
      Extended compute callback in postSolve. In this stage it's possible to fetch data from the
      merged body merge split algorithm containing external interactions with each merged body.
      \param postSolveData - the post solve data
      \param mergedBodyAlgorithm - the merged body merge split algorithm
      */
      virtual void onPostSolveComputeEx( const agxSDK::MergeSplitPostSolveData& postSolveData,
                                         const agxSDK::MergedBodyMergeSplitAlgorithm& mergedBodyAlgorithm );

      /**
      Push merge and/or split actions in simulation postSolve.
      \param postSolveData - the post solve data
      \param[out] actions - resulting actions
      */
      virtual void onPostSolveActions( const agxSDK::MergeSplitPostSolveData& postSolveData,
                                       agxSDK::MergeSplitActionContainer& actions );

      /**
      Stores internal data. This method has to be called explicitly by the derived class.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restores internal data. This method has to be called explicitly by the derived class.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

    private:
      friend class agxSDK::MergeSplitHandler;
      void setMergeSplitHandler( agxSDK::MergeSplitHandler* handler );

    private:
      MergeSplitHandler* m_mergeSplitHandler;
  };
}
