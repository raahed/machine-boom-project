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


#ifndef AGXHYDRAULICS_DETAIL_PISTON_ACTUATOR_CHAMBER_H
#define AGXHYDRAULICS_DETAIL_PISTON_ACTUATOR_CHAMBER_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/export.h>

namespace agxHydraulics
{
  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    A custom FlowUnit that represents the chambers inside a PistonActuator. The
    difference from a regular FlowUnit is that the PistonActuatorChamber will
    never add tank pressure to the output side since that side is pressed against
    eiher the piston or the bottom of the barrel.
    */
    class AGXHYDRAULICS_EXPORT PistonActuatorChamber : public agxHydraulics::FlowUnit
    {
      public:
        PistonActuatorChamber(agx::Real length, agx::Real area, agx::Real fluidDensity);

        /**
        Will only consider adding tank pressure to the input side.
        */
        virtual void addTankPressure() override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::PistonActuatorChamber);
        using agxHydraulics::FlowUnit::store;
        using agxHydraulics::FlowUnit::restore;

      protected:
        PistonActuatorChamber();
        virtual ~PistonActuatorChamber() {}
    };
  }
  /// \endcond
}

#endif
