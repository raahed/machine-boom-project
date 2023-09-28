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

#ifndef AGXHYDRAULICS_ACCUMULATOR_H
#define AGXHYDRAULICS_ACCUMULATOR_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/detail/AccumulatorConstraints.h>
#include <agxPowerLine/PowerLineConstraints.h>
#include <agxPowerLine/Connector.h>
#include <agx/Prismatic.h>
#include <agx/BallJoint.h>

namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(Accumulator);

  /**
  An accumulator is a component that stores fluid and pressure. The accumulator
  can be filled when the pressure and flow requirements of the system is low,
  and later emptied when the pumps are unable to provide sufficient pressure or
  flow.

  The accumulator is spring loaded, meaning that the more it has been filled
  the higher the pressure in the accumulator becomes.

  Some kind of mechanism is required to control the accumulator. A simple way
  is a user controlled stop valve.
  */
  class AGXHYDRAULICS_EXPORT Accumulator : public agxHydraulics::FlowUnit
  {
    public:
      /**
      Create a new Accumulator with the given dimensions and spring constant.
      The spring constant defines, toghether with the area, the ralationship
      between the current fluid volume and the pressure inside the Accumulator.
      */
      Accumulator(agx::Real length, agx::Real area, agx::Real fluidDensity, agx::Real springConstant);

      /**
      \return The spring constant that defines the relationship between fluid volume and displacement.
      */
      agx::Real getSpringConstant() const;

      /**
      Gives access to the constraint that models the internal behavior of the Accumulator.
      */
      const agxHydraulics::detail::AccumulatorConstraint* getAccumulatorConstraint() const;

    // Methods called by the rest of the power line framework.
    public:
#ifndef SWIG
      DOXYGEN_START_INTERNAL_BLOCK()
      virtual bool addNotification(agxSDK::Simulation* simulation) override;
      /** \internal */
      virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;
      DOXYGEN_END_INTERNAL_BLOCK()

      virtual agxPowerLine::DimensionAndSide getConnectableDimension(agxPowerLine::PhysicalDimension::Type type,
                                                                     agxPowerLine::Side side) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::Accumulator);
      using agxHydraulics::FlowUnit::store;
      using agxHydraulics::FlowUnit::restore;
#endif

    protected:
      Accumulator();
      virtual ~Accumulator();
      void createAccumulatorConstraint();

    private:
      agx::Real m_springConstant;
      agxHydraulics::detail::AccumulatorConstraintRef m_accumulatorConstraint;
  };
}


#endif
