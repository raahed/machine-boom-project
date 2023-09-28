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


#ifndef AGXHYDRAULICS_RELIEF_VALVE_H
#define AGXHYDRAULICS_RELIEF_VALVE_H


#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/NeedleValve.h>
#include <agxHydraulics/Pipe.h>

#include <agxHydraulics/detail/ReliefValveConstraints.h>

#include <agxPowerLine/TranslationalUnit.h>


namespace agx
{
  class Constraint;
}

namespace agxSDK
{
  class Simulation;
}



namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(ReliefValve);

  /**
  The relief valve is a hydraulic component that limits the pressure at some
  point in the system. It is configured by specifying a cracking pressure and a
  fully open pressure. When the pressure in the relief valve reaches the
  cracking pressure a separate drain to tank starts to open. As the pressure
  increases the drain continues to opens until the pressures reaches the fully
  open pressure. At that point the drain to tank will have reached the
  configured fully open area and the relief valve will be unable to prevent any
  further rise in pressure.

  The cracking pressure must always be less than the fully open pressure.

  The implementation uses a step function for the drain area. The drain is
  opened in stages as the pressure increases. The user can specify the number
  of steps, which is also called the number of drain pipes. More drain pipes
  allows the relief valve to regulate the pressure more precicely, reducing
  pressure fluctuations near step transitions, but comes with a performance
  penalty.

  At pressures below the cracking pressure the relief valve acts like a regular
  pipe. However, it is not a FlowUnit but instead a dimensionless Unit. This is
  because the relief valve has two separate flows: the input flow and the
  output flow. To represent this the ReliefValve contains two internal
  FlowUnits, called chambers, that represents the two flows.
  */
  class AGXHYDRAULICS_EXPORT ReliefValve : public agxPowerLine::Unit
  {
    public:
      /**
      Create a new relief valve.
      \param crackingPressure Pressure when the drain to tank begins to open.
      \param fullyOpenPressure Pressure when the drain to tank should reach it's maximum area.
      \param fullyOpenArea The area of the drain to tank when pressure it at or above the fullyOpenPressure.
      \param numDrains The number of drains to tank.
      */
      ReliefValve(agx::Real crackingPressure, agx::Real fullyOpenPressure, agx::Real fullyOpenArea, agx::UInt32 numDrains=6);

      /**
      Set the pressure range for the relief valve.

      This version is recommended over a call pair to setCrackingPressure and
      setFullyOpenPressure because it avoids the possibly invalid state that
      the first call in the pair might cause.
      \param crackingPressure The pressure when the drain to tank starts to open.
      \param fullyOpenPressure The pressure when the drain to tank is fully open.
      */
      void setPressures(agx::Real crackingPressure, agx::Real fullyOpenPressure);

      /**
      Set the pressure at which the drain should start to open.

      Must be less than the fully open pressure. It is recommended to use
      setPressures when changing both pressures to avoid invalid states.
      \see setPressures
      \param crackingPressure The pressure when the drain should start to open.
      */
      void setCrackingPressure(agx::Real crackingPressure);

      /**
      \return The pressure when the drain starts to open.
      */
      agx::Real getCrackingPressure() const;

      /**
      Set the pressure at which the drain should reach fully open.

      Must be greater than the cracking pressure. It is recommended to use
      setPressures when changing both pressures to avoid invalid states.
      \see setPressures
      \param fullyOpenPressure The pressure when the drain should reach fully open.
      */
      void setFullyOpenPressure(agx::Real fullyOpenPressure);

      /**
      \return The smallest pressure for which the drain is fully open.
      */
      agx::Real getFullyOpenPressure() const;

      /**
      \return The current pressure in the relief valve.
      */
      agx::Real getPressure() const;

      /**
      The opening fraction will always be a multiple of 1/num_drains, and will
      always be in the range [0..1], inclusive.
      \return The fraction that the drain has been opened.
      */
      agx::Real getOpeningFraction() const;

      /**
      \return The sum of the flow rates of all drains.
      */
      agx::Real getDrainFlowRate() const;

      /**
      \return An internal FlowUnit representing flow before the drain to tank.
      */
      agxHydraulics::FlowUnit* getInputChamber();
      const agxHydraulics::FlowUnit* getInputChamber() const;

      /**
      \return An internal FlowUnit representing flow after the drain to tank.
      */
      agxHydraulics::FlowUnit* getOutputChamber();
      const agxHydraulics::FlowUnit* getOutputChamber() const;

      /**
      \return The chamber at the given side, or nullptr if an invalid side is given.
      */
      agxHydraulics::FlowUnit* getChamber(agxPowerLine::Side side);
      const agxHydraulics::FlowUnit* getChamber(agxPowerLine::Side side) const;

      /**
      \return The number of steps in the drain area step function.
      */
      agx::UInt32 getNumDrains() const;


      /**
      \param index The drain pipe to return. Must be less than the number of drains.
      \return The drain pipe at the given index.
      */
      agxHydraulics::Pipe* getDrain(agx::UInt32 index);

      /**
      \param index The drain pipe to return. Must be less than the number of drains.
      \return The drain pipe at the given index.
      */
      const agxHydraulics::Pipe* getDrain(agx::UInt32 index) const;

      /**
      The implementation uses constraints to control when drain pipes should
      open or close.
      \param index The drain pipe constraint to return. Must be less than the number of drains.
      \return The constraint controlling drain pipe at the given index.
      */
      agx::Constraint* getFlowBlockConstraint(agx::UInt32 index);

      /**
      \return The FlowConnector which connects the internal flow units.
      */
      agxHydraulics::FlowConnector* getDrainConnector();


    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
#ifndef SWIG
      /**
      Always only the flow type.
      */
      virtual void getConnectableDimensionTypes(
          agxPowerLine::PhysicalDimension::TypeVector& types, agxPowerLine::Side side) const override;

      /**
      For flow, returns either the input chamber or the output chamber depending
      on the given side. Returns nothing for any other dimension type.
      \return The PhysicalDimension and side that that other Units may connect to.
      */
      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
          agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side) override;

      /**
      Called when the ReliefValve is added to the simulation. Does some internal
      constraint setup.
      */
      virtual bool addNotification(agxSDK::Simulation *simulation) override;

      /**
      Called when the relief valve is removed from the simulation. Does some
      internal constraint teardown.
      */
      virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation *simulation) override;

      /**
      Called by the PowerLine during stream serialization. Stores internal data
      into the given stream.
      \param str - The StorageStream to store internal data to.
      \return True if the store was successful. False otherwise.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      virtual bool postStore(agxStream::StorageStream& str) const override;

      /**
      Called by the PowerLine durint stream deserialization. Restores internal
      data from the given stream.
      \param str - The StorageStream to restore internal data from.
      \param True if the restore was successful. False otherwise.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      virtual bool postRestore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::ReliefValve);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      ReliefValve();

    private:
      agxHydraulics::PipeRef m_inputChamber;
      agxHydraulics::PipeRef m_outputChamber;
      agxHydraulics::PipeRefVector m_drains;
      agxHydraulics::detail::LimitedFlowBlockConstraintRefVector m_flowBlockConstraints;

      agx::Real m_crackingPressure;
      agx::Real m_fullyOpenPressure;
      agx::Real m_fullyOpenArea;

      // Owned by the base class SubGraph.
      FlowConnectorObserver m_drainConnector;
  };
}

#endif


