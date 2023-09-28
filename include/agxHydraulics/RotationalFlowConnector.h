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


#ifndef AGXHYDRAULICS_ROTATIONAL_FLOW_CONNECTOR_H
#define AGXHYDRAULICS_ROTATIONAL_FLOW_CONNECTOR_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/PressureConnector.h>

#include <agxPowerLine/RotationalDimension.h>

namespace agxHydraulics
{
  /**
  Abstract base class containing overlapping functionality for pump and motor.
  */
  class AGXHYDRAULICS_EXPORT RotationalFlowConnector : public agxHydraulics::PressureConnector
  {
    public:
      /**
      Set the displacement of the pump or motor. The displacement defines the
      conversion ratio between rotation and flow, and is always measured in
      units of flow per unit of rotation.

      \param displacement - The new displacement of the pump or motor.
      */
      virtual void setDisplacement(agx::Real displacement);

      /**
      \return The displacement of the pump or motor.
      */
      virtual agx::Real getDisplacement() const;

      /**
      Set the compliance of the constraint. Increasing the compliance will
      increase the leakage of the pump or motor.

      The unit of the compliance is volume per time per pressure, or leakage
      rate per pressure.
      */
      void setCompliance(agx::Real compliance);

      /**
      \return The compliance of the constraint, a.k.a. the leakage parameter of the pump or motor.
      */
      agx::Real getCompliance() const;

      /**
      Get the pressure that the pump or motor generates. This is not the same
      as the pressure in the pump or motor.

      \return The current pressure that the pump or motor generates.
      */
      agx::Real getPressure() const override;



    // Methods called by the rest of the power line framework.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
           const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

#ifndef SWIG
      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxHydraulics::RotationalFlowConnector);
      void restore(agxStream::InputArchive& in) override;
      void store(agxStream::OutputArchive& out) const override;
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      RotationalFlowConnector();
      virtual ~RotationalFlowConnector() {}

      virtual agx::Real displacementToRatio(agx::Real displacement) = 0;
      virtual agx::Real constraintForceToPressure(agx::Real constraintForce, agx::Real displacement) const = 0;
      virtual agx::Real pressureToConstraintForce(agx::Real pressure, agx::Real ratio) const = 0;

    protected:
      agx::Real m_displacement;
      agx::Real m_compliance;
  };

  typedef agx::ref_ptr<agxHydraulics::RotationalFlowConnector> RotationalFlowConnectorRef;
}

#endif
