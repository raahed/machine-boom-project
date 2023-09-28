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



#ifndef AGXHYDRAULCIS_DETAIL_PISTON_ACTUATOR_CONNECTOR_H
#define AGXHYDRAULCIS_DETAIL_PISTON_ACTUATOR_CONNECTOR_H

#include <agxPowerLine/ActuatorConnector.h>
#include <agxHydraulics/export.h>

namespace agxHydraulics
{
  class PistonActuator;

  AGX_DECLARE_POINTER_TYPES(PistonActuatorConnector);

  /**
  The PistonChamberConnector is an internal Connector to the PistonActuator. It
  defines the relationship between the relative positions of the bodies and the
  amount of fluid in one of the chambers. One PistonChamberConnector is created
  for each chamber in the PistonActuator.
  */
  class AGXHYDRAULICS_EXPORT PistonActuatorConnector : public agxPowerLine::ActuatorConnector
  {
    public:
      PistonActuatorConnector(agxHydraulics::PistonActuator* actuator);

      /**
      \return The area of the chamber that the PistonChamberConnector connects to.
      */
      virtual agx::Real getArea() const;

      /**
      \return The PistonActuator that this connector is part of.
      */
      agxHydraulics::PistonActuator* getPistonActuator();



    // Methods called by the rest of the power line framework.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      */
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

      /**
      */
      virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
          const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

      /**
      \return The difference between the volume of the chamber and the volume
              of fluid currently in the chamber. High violation means chamber
              larger than fluid.
      */
      virtual agx::Real calculateViolation() const override;

      /**
      \return True for the PistonChamberConnector connected to the output chamber.
              False for the PistonChamberConnector connected to the input chamber.
      */
      virtual bool getReverseOrder() const override;

#ifndef SWIG
      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::PistonActuatorConnector);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      PistonActuatorConnector();
      virtual ~PistonActuatorConnector();

      agx::observer_ptr<agxHydraulics::PistonActuator> m_pistonActuator;
  };
}

#endif
