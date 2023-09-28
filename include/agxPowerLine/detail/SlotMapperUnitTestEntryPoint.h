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


#ifndef AGXPOWERLINE_SLOT_MAPPER_UNIT_TEST_ENTRY_POINT_H
#define AGXPOWERLINE_SLOT_MAPPER_UNIT_TEST_ENTRY_POINT_H

#include <agxPowerLine/SlotMapper.h>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4512) // warning C4512:  new behavior: assignment operator could not be generated
#endif

namespace agxPowerLine
{
  namespace detail
  {

    /**
    The UnitTestEntryPoint is a backdoor into the SlotMapper's internal state
    used by the unit tests to ensure that the SlotMapper implementation is
    correct. Should not be used for any other purpose.
    */
    class AGXMODEL_EXPORT SlotMapperUnitTestEntryPoint
    {
      public:
        SlotMapperUnitTestEntryPoint(SlotMapper& mapper);

        bool containsInFull(agxPowerLine::detail::Rotational1DofState* state);

        bool containsInNonfull(agxPowerLine::detail::Rotational1DofState* state);

        size_t getNumFullPacks() const;

        size_t getNumNonfullPacks() const;

        int getNumDimensionsIn(agx::RigidBody* body);

      private:
        SlotMapper& m_mapper;
    };
  }
}

#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif
