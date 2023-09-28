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

#ifndef AGXHYDRAULICS_FLOW_UNIT_H
#define AGXHYDRAULICS_FLOW_UNIT_H

#include <agxHydraulics/FlowConnector.h>
#include <agxHydraulics/FlowDimension.h>

#include <agxPowerLine/Unit.h>
#include <agxPowerLine/Sides.h>

namespace agx
{
  class Constraint;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxModel
{
  class DriveTrain;
}




namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(FlowUnit);
  AGX_DECLARE_VECTOR_TYPES(FlowUnit);


  /**
  FlowUnit is the base class for Units that contains a FlowDimension. It is
  tightly coupled with the FlowConnector, which is used to connect FlowUnits
  together.

  A FlowUnit has a single input end and a single output end. A single
  FlowConnector can be connected to each end. An end that isn't attached to
  anything is assumed to be attached to the tank. Other PressureConnectors, the
  Pump for example, does not consume an end.

  FlowUnits have inertia and flow resistance that is calculated from the pipe
  geometry and fluid properties. The inertia is a measure of how hard it is to
  produce flow through the FlowUnit, while flow resistance is a measure of how
  hard it is to maintain the flow rate. The flow resistance increases with flow
  rate.
  */
  class AGXHYDRAULICS_EXPORT FlowUnit : public agxPowerLine::Unit
  {
    public:
      /**
      Create a FlowUnit with inertia calculated from the given parameters.
      \param length - Length of the FlowUnit
      \param area - Area of the flow unity
      \param fluidDensity - Density of the fluid
      */
      FlowUnit(agx::Real length, agx::Real area, agx::Real fluidDensity);

      DOXYGEN_START_INTERNAL_BLOCK()
      FlowUnit(agx::Real length, agx::Real area, agx::Real fluidDensity,
               agx::RigidBody* body, agx::UInt8 elementIndex);

      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      Create a FlowUnit with the given inertia. The length, area, and density
      attributes will be chosen so that the requested inertia is produced.
      */
      explicit FlowUnit(agx::Real inertia);

      /**
      Set the length of the FlowUnit. Increasing the length will increase both
      the inertia and flow resistance.
      */
      void setLength(agx::Real length);

      /**
      \return the length of the FlowUnit
      */
      agx::Real getLength() const;

      /**
      Set the area of the FlowUnit. Increasing the area will decrease both the
      inertia and the flow resistance.
      \return false if area is <= 0
      */
      bool setArea(agx::Real area);

      /**
      \return The area of the FlowUnit
      */
      agx::Real getArea() const;

      /**
      Set the density of the fluid. Increasing the density will increase both
      the inertia and the flow resistance.
      */
      void setFluidDensity(agx::Real density);

      /**
      \return the fluid density used in the FlowUnit
      */
      agx::Real getFluidDensity() const;

      /**
      The viscosity specified here is the kinematic viscosity, measured in
      area per time. The default viscosity is 1e-5.

      Increasing the viscosity increases the flow resistance. Zero viscosity
      produces zero flow resistance.
      */
      void setFluidViscosity(agx::Real viscosity);

      /**
      \return the viscosity used in the FlowUnit
      */
      agx::Real getFluidViscosity() const;

      /**
      \return The current flow rate through the FlowUnit, measured in units
      of volume per time.
      */
      agx::Real getFlowRate() const;

      /**
      Add extra pressure to the FlowUnit at the given side. A high pressure at
      the input side will act to generate positive flow through the FlowUnit,
      while pressure at the output side will act to generate negative flow.

      \param pressure - The pressure to add.
      \param side - The side of the FlowUnit where the pressure should be applied.
      */
      void addPressure(agx::Real pressure, agxPowerLine::Side side);

      /**
      Measure the pressure at the given side of the FlowUnit. Will only
      include pressure from a FlowConnector, so pressure added by other
      Connectors (e.g. pump/motor) or FlowUnit::addPressure will not be
      included.

      If the side is unconnected then the tank pressure will be returned.

      \param side The side to measure pressure at.
      \return The pressure at the given side.
      */
      agx::Real getPressure(agxPowerLine::Side side) const;

      /**
      Measure the pressure at the inlet of the FlowUnit.
      \see getPressure
      \return The pressure at the input FlowConnector, or tank pressure.
      */
      agx::Real getInputPressure() const;

      /**
      \deprecated Use getInputPressure instead.
      \see getInputPressure.
      */
      agx::Real getInletPressure() const;

      /**
      Measure the pressure at the outlet of the FlowUnit.
      \see getPressure
      \return  The pressure at the outlet FlowConnector, or tank pressure.
      */
      agx::Real getOutputPressure() const;

      /**
      \deprecated Use getOutputPressure instead.
      \see getOutputPressure
      */
      agx::Real getOutletPressure() const;

      /**
      \return The FlowDimension held by this FlowUnit.
      */
      agxHydraulics::FlowDimension* getFlowDimension();

      /**
      \return The FlowDimension held by this FlowUnit.
      */
      const agxHydraulics::FlowDimension* getFlowDimension() const;


      /**
      \return the FlowConnector that the given pipe end is attached to. Will
      be nullptr if that end is not yet connected to another FlowUnit.
      */
      virtual agxHydraulics::FlowConnector* getFlowConnector(agxPowerLine::Side side);

      /**
      \return the FlowConnector that the given pipe end is attached to. Will
      be nullptr if that end is not yet connected to another FlowUnit.
      */
      virtual const agxHydraulics::FlowConnector* getFlowConnector(agxPowerLine::Side side) const;

      /**
      \return the FlowConnector that the input side is attached to. Will be
      nullptr if the input is not yet connected to another FlowUnit.
      */
      agxHydraulics::FlowConnector* getInputFlowConnector();

      /**
      \return the FlowConnector that the input side is attached to. Will be
      nullptr if the input is not yet connected to another FlowUnit.
      */
      const agxHydraulics::FlowConnector* getInputFlowConnector() const;

      /**
      \return the FlowConnector that the output side is attached to. Will be
      nullptr if the output is not yet connected to another FlowUnit.
      */
      agxHydraulics::FlowConnector* getOutputFlowConnector();

      /**
      \return the FlowConnector that the output side is attached to. Will be
      nullptr if the output is not yet connected to another FlowUnit.
      */
      const agxHydraulics::FlowConnector* getOutputFlowConnector() const;

      /**
      \return true if there is a FlowConnector attached at the given end.
      False otherwise.
       */
      bool isConnected(agxPowerLine::Side end) const;

      using Unit::isConnected;

      /**
      \return The FlowConnector that connects the two FlowUnits. Returns nullptr if
      there is none.
      */
      static FlowConnector* findInCommonFlowConnector(FlowUnit* a, FlowUnit* b);


    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Computes that AGX linear velocity damping term that corresponds to the
      given flow rate through this FlowUnit. The base implementation assumes
      laminar flow at all times, but subclasses may override this behavior.
      */
      virtual agx::Real computeFrictionLoss(agx::Real flowRate);

      /**
      Add tank pressure, supplied by agxHydraulics::Hydraulics. The default
      implementation will add tank pressure to all sides that doesn't have a
      connected FlowConnector. Subclasses may override addTankPressure in order
      to prevent tank pressure on internal FlowUnits.
      */
      virtual void addTankPressure();

#ifndef SWIG
      /**
      Called by the power line during step preparation.
      */
      virtual bool preUpdate(agx::Real timeStep) override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::FlowUnit);
#endif

    protected:
      /**
      Used only by store/restore.
      */
      FlowUnit();
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      void init();

      void updateBodyMass();

      agx::Real computeReynoldsNumber() const;
      static agx::Real computeReynoldsNumber(agx::Real area, agx::Real flowRate, agx::Real viscosity);
      agx::Real computeFrictionFactor() const;
      agx::Real computeLaminarFrictionalLoss(agx::Real flowRate) const;
      agx::Real computeQuadraticFrictionalLoss(agx::Real flowCoefficient) const;
      agx::Real updateFrictionLoss();

    protected:
      virtual ~FlowUnit() {}

    protected:
      agxHydraulics::FlowDimensionRef m_flowDimension;

    private:
      agx::Real m_length;
      agx::Real m_area;

      /// \todo Who should own this number?
      agx::Real m_fluidDensity;

      /// \todo Who should own this number?
      agx::Real m_viscosity;
  };

}

#endif
