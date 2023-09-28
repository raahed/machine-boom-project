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

#ifndef AGXPOWERLINE_SLOT_MAPPER_H
#define AGXPOWERLINE_SLOT_MAPPER_H

#include <agxModel/export.h>
#include <agx/Vector.h>
#include <agx/Referenced.h>
#include <agxPowerLine/detail/DimensionState.h>
#include <agxPowerLine/detail/PackingInfo.h>



namespace agxPowerLine
{
  class PowerLine;

  namespace detail
  {
    class SlotMapperUnitTestEntryPoint;
  }
}


namespace agxPowerLine
{
  /**
  The SlotMapper is reponsible for mapping one dimensional dimension states to
  slots in the RigidBodies. Whenever a PowerLine is given a PhysicalDimension
  with a packable one dimensional dimension state the state is passed on to the
  slot mapper who finds a suitable RigidBody to pack the new state into.

  Some of the body handling methods on DimensionState are routed through a slot
  mapper in order to keep the packing information held by the slot mappter up to
  date.
  */
  class AGXMODEL_EXPORT SlotMapper : public agx::Referenced, public agxStream::Serializable
  {
    public:
      SlotMapper(agxPowerLine::PowerLine* powerLine);

      /**
      Pack means move the state to a body that SlotMapper knows about. A new body
      may be created in the process.
      */
      void pack(agxPowerLine::detail::Rotational1DofState* dimension);
      void pack(agxPowerLine::detail::Translational1DofState* dimension);


      /**
      After a call to unpack the given dimension will not share a body with any
      other dimension and will be forgotten by the SlotMapper. Call \p pack again
      to re-enable packing.
      */
      void unpack(agxPowerLine::detail::Rotational1DofState* dimension);
      void unpack(agxPowerLine::detail::Translational1DofState* dimension);

      /**
      Move the dimension state from the dimensions' current body to a new body
      and mark the dimension as not packable.

      Since the dimension is no longer packable, the SlotMapper will no longer
      keep track of it ano no further slot mapper operation should be done on it.
      */
      void reserve(agxPowerLine::detail::Rotational1DofState* dimension);
      void reserve(agxPowerLine::detail::Translational1DofState* dimension);

      /**
      Move the dimension state from the dimensions' current body to a new one.
      The dimension will still be packable.
      */
      void split(agxPowerLine::detail::Rotational1DofState* dimension);
      void split(agxPowerLine::detail::Translational1DofState* dimension);

      /*
      Move the given dimension state from its current body and slot to the given
      slot in the given body.
      */
      void move(agxPowerLine::detail::Rotational1DofState* dimension, agx::RigidBody* body, agx::UInt8 slot);
      void move(agxPowerLine::detail::Translational1DofState* dimension, agx::RigidBody* body, agx::UInt8 slot);


      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::SlotMapper);

    private:
      SlotMapper(); // For serialization only.
      void assertValid() const;

    private:
      /**
      The UnitTestEntryPoint is a backdoor into the SlotMapper's internal state
      used by the unit tests to ensure that the SlopMapper implementation is
      correct. Should not be used for any other purpose.
      */
      friend class agxPowerLine::detail::SlotMapperUnitTestEntryPoint;

    private:
      agxPowerLine::detail::PackingInfos m_fullPackings; ///<! List of RigidBodies in which all slots are filled.
      agxPowerLine::detail::PackingInfos m_nonfullPackings; ///<! List of RigidBodies in which there are free slots.
      agxPowerLine::PowerLine* m_powerLine;
  };

  AGX_DECLARE_POINTER_TYPES(SlotMapper);
}

#endif
