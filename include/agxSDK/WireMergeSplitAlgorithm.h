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

#include <agx/Kernel.h>

namespace agxWire
{
  class Wire;
  class Node;
  class WireWinchController;
}

namespace agxSDK
{
  class AGXPHYSICS_EXPORT WireMergeSplitAlgorithm : public agxSDK::MergeSplitAlgorithm
  {
    public:
      /**
      Callback when wires recives calls to attach. If this algorithm is active, this
      method will split the wire if it has been merged with this algorithm.
      \param wire - called wire
      \param node - node to attach
      \param begin - true if \p node is being attached to begin of the wire, false end of the wire
      */
      static void onAttach( agxWire::Wire* wire, agxWire::Node* node, agx::Bool begin );

      /**
      Callback when wires recives calls to attach (given winch). If this algorithm is active, this
      method will split the wire if it has been merged with this algorithm.
      \param wire - called wire
      \param winch - winch to attach
      \param begin - true if \p winch is being attached to begin of the wire, false end of the wire
      */
      static void onAttach( agxWire::Wire* wire, agxWire::WireWinchController* winch, agx::Bool begin );

      /**
      Callback when wires recives calls to detach. If this algorithm is active, this
      method will split the affected lumped node if it has been merged with this algorithm.
      \param wire - called wire
      \param begin - true if detach of begin of the wire, false end of the wire
      */
      static void onDetach( agxWire::Wire* wire, agx::Bool begin );

    public:
      struct PostSolveData
      {
        PostSolveData()
          : actions( 32u )
        {
        }

        MergeSplitActionContainer actions;
      };
      using PostSolveComputeDataContainer = agx::Vector<PostSolveData>;

    public:
      WireMergeSplitAlgorithm();

      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::WireMergeSplitAlgorithm );

    protected:
      virtual ~WireMergeSplitAlgorithm();

      virtual void onPreStepActions( MergeSplitActionContainer& actions ) override;
      virtual void onPostSolveComputeEx( const MergeSplitPostSolveData& postSolveData,
                                         const MergedBodyMergeSplitAlgorithm& mergedBodyAlgorithm ) override;
      virtual void onPostSolveActions( const MergeSplitPostSolveData& postSolveData, MergeSplitActionContainer& actions ) override;

    private:
      agx::LambdaKernelRef m_postSolveKernel;
  };
}
