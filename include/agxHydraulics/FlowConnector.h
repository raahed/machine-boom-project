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



#ifndef AGXHYDRAULICS_FLOW_CONNECTOR_H
#define AGXHYDRAULICS_FLOW_CONNECTOR_H

#include <agxHydraulics/export.h>
#include <agxHydraulics/PressureConnector.h>
#include <agxPowerLine/PowerLineUtils.h>


namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(FlowConnector);
  AGX_DECLARE_VECTOR_TYPES(FlowConnector);

  /**
  A FlowConnector is a junction of a set of FlowUnits. The FlowConnector
  ensures that the sum of flows through the junction is close to zero, i.e.,
  that no fluid is created or destroyed. Regularization in the constraints
  causes a small amount of fluid to either be stored in the junction or lost,
  so output flow rate is not always exactly equal to the input flow rate.
  */
  class AGXHYDRAULICS_EXPORT FlowConnector : public agxHydraulics::PressureConnector
  {
    public:
      /**
      A flow connector can either be holonomic or nonholonomic. A holonomic
      FlowConnector tracks the total input and output flow and flexes in order
      to hold any stored fluid. A nonholonomic FlowConnector does not track the
      differences in flow rate. A higher input flow rate than output flow rate
      will produce a leak.
       */
      FlowConnector(bool holonomic = true);

      /**
      Connect the given unit to this FlowConnector.

      If the given side already has a FlowConnector then that FlowConnector will
      be merged into 'this'. The old FlowConnector will be empty.
      */
      virtual bool connect(
        agxPowerLine::Side mySide,
        agxPowerLine::Side unitSide,
        agxPowerLine::Unit* unit) override;

      using agxHydraulics::PressureConnector::connect;

      /**
      Set the compliance of the flow constraint.

      For a holonomic FlowConnector this controls the flexibility of the
      junction. A larger compliance causes the FlowConnector to hold more
      fluid for a given pressure. The unit is volume/pressure. At steady
      state the amount of fluid held by the junction is

          volume = compliance * pressure

      For a nonholonimc constraint the compliance controls the amount of
      leakage at the junction. A larger compliance causes the FlowConnector
      to leak more fluid for a given pressure. The unit is volume/time / pressure.
      At steady state the amount of fluid leaked by the junction is

         volume / time = compliance * pressure
      */
      void setCompliance(agx::Real volumePerPressure);

      /**
      \return The compliance of the flow constraint.
      \see setCompliance
      */
      agx::Real getCompliance() const;

      /**
      Set the damping of the flow constraint. Has no effect on nonholonomic
      FlowConnectors.
       */
      void setDamping(agx::Real damping);

      /**
      \return The damping of the flow constraint.
      */
      agx::Real getDamping() const;

      /// Alias for setCompliance(agx::Real).
      void setVolumeExpansionPressureRatio(agx::Real ratio);

      /// Alias for getCompliance().
      agx::Real getVolumeExpansionPressureRatio() const;

      /**
      \return The current pressure in the junction.
       */
      virtual agx::Real getPressure() const override;

      /**
      Only valid for holonomic FlowConnectors.
      \return The amount of fluid currently stored in the junction.
       */
      agx::Real getStoredFluid() const;




      /**
      Set the default compliance for new flow connectors. This will only change
      the initial compliance for flow connectors created after the call to
      setDefaultCompliance. Flow connectors already created remain unchanged.

      \see FlowConnector::setCompliance(agx::Real)
       */
      static void setDefaultCompliance(agx::Real compliance);

      /**
      \return The compliance that new FlowConnectors get upon creation.
      */
      static agx::Real getDefaultCompliance();

      /**
      Set the default damping for new flow connectors. This will only change
      the initial damping for flow connectors created after the call to
      setDefaultDamping. Flow connectors already created remain unchanged.
       *
      \see FlowConnector::setDamping(agx::Real)
       */
      static void setDefaultDamping(agx::Real damping);

      /**
      \return The damping that new FlowConnectors get upon creation.
      */
      static agx::Real getDefaultDamping();


    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:

      DOXYGEN_START_INTERNAL_BLOCK()
      void setStoredFluid(agx::Real storedFluid);


      /**
      Called by the PowerLine during initialization.
      */
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;


      /**
      Called by the power line during step preparation.
      */
      virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
          const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

      /**
      Mark the flow constraint as impacting.
      */
      void setIsImpacting(bool impact);

#ifndef SWIG
      /**
      Called by the PowerLine. Store constraint related data to the given
      StorageStream.
      */
      virtual bool postStore(agxStream::StorageStream& str) const override;

      /**
      Called by the PowerLine. Restore constraint data from the given
      StorageStream.
      */
      virtual bool postRestore(agxStream::StorageStream& str) override;


      /**
      Called by the PowerLine during store. Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Called by the PowerLine during restore. Restores internal data from
      stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;


      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::FlowConnector);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

// Swig tries to generate bindings to this protected connect member function
// which leads to compiler errors since the function is protected.
#ifndef SWIG
      virtual bool connect(
          agxPowerLine::Unit* inputUnit, agxPowerLine::Side inputUnitSide,
          agxPowerLine::Side outputUnitSide, agxPowerLine::Unit* outputUnit) override;
#endif

      virtual ~FlowConnector() {}

    private:
      agx::Real m_volumeExpansionPressureRatio;
      agx::Real m_damping;
      bool m_holonomic;

      static agx::Real s_defaultCompliance;
      static agx::Real s_defaultDamping;
  };


}

#endif
