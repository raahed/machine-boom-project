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

#include <agxCable/SegmentDamageState.h>
#include <agxCable/CableDamageStateFilter.h>
#include <agxCable/Cable.h>

#include <agxStream/Serializable.h>


namespace agxCable
{
  class SegmentDamageState;
}

namespace agxCable
{
  /**
  Inspect a cable's current state and compute a SegmentDamageState for each
  segment. The SegmendDamageStates are later used by CableDamage instances to
  estimate cable damage.

  A filter is used to select which parts of the SegmentDamageStates that are
  computed. This makes it possible to implement other damage estimation models
  in the future.
  */
  class AGXCABLE_EXPORT CableDamageState : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Create a new CableDamageState and register it with the given cable.
      SegmentDamageStates will be updated ever time step.

      \param cable - The cable for which SegmendDamageStates are to be computed.
      */
      CableDamageState(agxCable::Cable* cable);

      /**
      Provides read-only access to the segment damage states.

      \return A collection of SegmentDamageStates.
      */
      const agxCable::SegmentDamageStateVector& getSegmentDamageData() const;

      /**
      \return The number of SegmentDamageStates tracked by this CableDamageState.
      */
      size_t getNumSegmentDamageStates() const;

      /**
      Set the filter used to determine which parts of the SegmentDamageStates
      that should be computed.

      \param filter - Filter specification.
      */
      void setCableDamageStateFilter(agxCable::CableDamageStateFilter filter);

      /**
      \return The filter used when determining which parts of the SegmentDamageStates that should be computed.
      */
      agxCable::CableDamageStateFilter getCableDamageStateFilter() const;


      /**
      Enable or disable computation of a single piece of the SegmentDamageState.

      \param damageType - The piece of the SegmentDamageState that should be enabled or disabled.
      \param enable - True of the given state type is to be computed. False otherwise.
      */
      void setEnable(agxCable::DamageStateTypes::StateType damageType, bool enable);

      /**
      Test if the given piece of the SegmentDamageState is part of the set of state that is computed.

      \param damageType - The piece of the SegmentDamageState to test.
      \return True if the state is computed. False otherwise.
      */
      bool isEnabled(agxCable::DamageStateTypes::StateType damageType) const;

      agxCable::Cable* getCable();
      const agxCable::Cable* getCable() const;

    public: // Methods called by the AGX framework.
      virtual void pre();
      virtual void post();

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::CableDamageState);

      DOXYGEN_START_INTERNAL_BLOCK()
      CableDamageState();
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      virtual ~CableDamageState();

    private:
      CableObserver m_cable;
      SegmentDamageStateVector m_segmentDamages;
      CableDamageStateFilter m_filter;
  };

  AGX_DECLARE_POINTER_TYPES(CableDamageState);
}
