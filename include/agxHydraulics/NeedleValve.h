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


#ifndef AGXHYDRAULICS_NEEDLE_VALVE_H
#define AGXHYDRAULICS_NEEDLE_VALVE_H

#include <agxHydraulics/FlowUnit.h>


namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(NeedleValve);

  /**
  A needle valve represents a controllable orifice opening. Reducing the
  orifice area increases the resistance to change in flow rate (inertia)
  through the valve, but also increases a viscous velocity damping parameter,
  i.e., increases the frictional loss in the valve. This makes it possible to
  control the flow rate through the valve for a given pressure difference over
  the valve.

  Very narrow needle valve openings produces hard to solve numerical systems
  and may lead to instabilities in the simulation. There is a cutoff area where
  the valve is considered to be shut. If making the orifice very small brings
  instabilities to the simulation, then increase the cutoff area. If very
  narrow orifices are required, then this cutoff can be lowered.
  */
  class AGXHYDRAULICS_EXPORT NeedleValve : public agxHydraulics::FlowUnit
  {
    public:
      /**
      Create a new needle valve with the given area and fluid density. The
      valve is initially fully open.
      */
      NeedleValve(agx::Real maxOpeningArea, agx::Real fluidDensity);

      /**
      Set the orifice opening area using an argument in the range [0..1] that
      repsents areas between 0 and maxOpeningArea. The value will be clamped to
      this range and a resulting area less than the cutoff area will be
      flushed to zero.
      */
      void setOpeningFraction(agx::Real fractionOfMax);

      /**
      Set the orifice opening diameter using an argument in the range [0..1]
      that represents diameters between closed and fully open. The value will
      be clamped to this range and a resulting area leass than the cutoff area
      will be flushed to zero.
      */
      void setOpeningDiameterFraction(agx::Real fractionOfMax);

      /**
      Set the area of the valve directly. The argument will be clamped to the
      range [0..maxOpeningArea] and values less than the cutoff area will be
      flushed to zero.
      */
      void setOpeningArea(agx::Real area);

      /**
      \return The current area opening fraction, a number in the range [0..1].
      */
      agx::Real getOpeningFraction() const;

      /**
      \return The current diameter opening fraction, a number in the range [0..1].
      */
      agx::Real getOpeningDiameterFraction() const;

      /**
      \return The current opening area, a number in the range [0..maxOpeningArea].
      */
      agx::Real getOpeningArea() const;

      /**
      \return The maximum that area that the value can be opened to.
      */
      agx::Real getMaxOpeningArea() const;

      /**
      Set the flow coefficient of the valve. A larger value lets the fluid pass
      through the valve more easily, and a smaller values causes a larger
      pressure drop over the valve.
      */
      void setFlowCoefficient(agx::Real coefficient);

      /**
      \return The flow coefficient of the valve.
      */
      agx::Real getFlowCoefficient() const;

      /**
      Set smallest area that the needle valve is allowed to have. Any request to
      set the area smaller than this will shut the valve.
      */
      void setCutoffArea(agx::Real cutoff);

      /**
      \return The smallest possible non-zero area of the needle valve.
      */
      agx::Real getCutoffArea() const;


    // Methods called by the rest of the power line framework.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agx::Real computeFrictionLoss(agx::Real flowRate) override;
      DOXYGEN_END_INTERNAL_BLOCK()

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::NeedleValve);

      using agxHydraulics::FlowUnit::store;
      using agxHydraulics::FlowUnit::restore;
#endif
    protected:
      NeedleValve();
      virtual ~NeedleValve() {}

      // Just to get rid of warning C4512
      NeedleValve& operator=(const NeedleValve&);

    private:
      agx::Real m_maxOpeningArea;
      agx::Real m_flowCoefficient;
      agx::Real m_cutoffArea;
      agx::Real m_minFlowRateForLossCalculation;
      bool m_closed;
  };
}



#endif
