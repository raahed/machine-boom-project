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

#include <agxModel/export.h>

#include <agxSDK/LinkedStructureComponent.h>
#include <agxSDK/ContactEventListener.h>

#include <agx/BitState.h>

namespace agxModel
{
  /**
  Beam component manipulating contacts with the beam, preventing contact
  normals from being along the beam, e.g., contact points generated between
  two beam segments. Such contact normals are either transformed to be
  completely orthogonal to the beam axis or disabled if other contacts are
  considered valid. To enable this feature, add this component to a beam:
      beam->addComponent( new agxModel::BeamContactHandler() );
  */
  class AGXMODEL_EXPORT BeamContactHandler : public agxSDK::LinkedStructureComponent
  {
    public:
      /**
      Default constructor using dot threshold: 0.05.
      */
      BeamContactHandler();

      /**
      Set projection threshold of a contact normal along the beam which
      triggers the algorithm that modifies the contact normal to be completely
      orthogonal to the beam axis. I.e., the contact normal is considered valid
      if:
          abs( beam_axis * contact_normal ) < threshold
      This threshold should be close to zero, default: 0.05.
      \param dotThreshold - new threshold > 0
      */
      void setDotThreshold( agx::Real dotThreshold );

      /**
      \return the currently used beam axis dot contact normal threshold that triggers
              the algorithm which manipulates the contact
      */
      agx::Real getDotThreshold() const;

      /**
      Enable/disable debug rendering of contact points handled by this component.
      \param enable - true to enable, false to disable (Default: Disabled)
      */
      void setEnableDebugRendering( agx::Bool enable );

      /**
      \return true if debug rendering is enabled, otherwise false
      */
      agx::Bool getEnableDebugRendering() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::BeamContactHandler );

    protected:
      virtual ~BeamContactHandler();

      virtual void onAddNotification( agxSDK::Simulation* simulation ) override;
      virtual void onRemoveNotification( agxSDK::Simulation* simulation ) override;
      virtual void onPreStep() override;

      using agxSDK::LinkedStructureComponent::onAddNotification;
      using agxSDK::LinkedStructureComponent::onRemoveNotification;

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      enum StateEnum : agx::UInt32
      {
        DEBUG_RENDER = 1 << 0,
      };
      using State = agx::BitState<StateEnum, agx::UInt32>;

    private:
      agxSDK::ContactEventListenerRef m_listener;
      agx::Real m_dotThreshold;
      State m_state;
  };
}
