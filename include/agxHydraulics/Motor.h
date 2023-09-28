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


#ifndef AGXHYDRAULICS_MOTOR_H
#define AGXHYDRAULICS_MOTOR_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/RotationalFlowConnector.h>

#include <agxPowerLine/RotationalDimension.h>

namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(Motor);

  /**
  A PowerLine Connector that connects a flow input to a rotational output. Flow
  at the input is converted to rotation at the output according to the
  configured displacement. The displacement is measured in terms of volume per
  unit of rotation, measured in radians. For simulations using the SI unit
  system, a motor displacement of 1.0 will rotate the rotational dimension one
  radian for every cubic meter of flud flow through the motor. 2*pi cubic
  meters of fluid is required for a full revolution. Displacements are often
  much smaller than this.
  */
  class AGXHYDRAULICS_EXPORT Motor : public agxHydraulics::RotationalFlowConnector
  {
    public:
      /**
      Create a new MotorConnector with displacement 1.
      */
      Motor();

      /**
      Create a new MotorConnector with the specified displacement.
      \param displacement - The rotation-to-flow conversion factor.
      */
      explicit Motor(agx::Real displacement);

    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

      DOXYGEN_START_INTERNAL_BLOCK()
#ifndef SWIG
      /**
      Called by the PowerLine during stream serialization. Stores internal data
      into the given stream.
      \param str - The StorageStream to store internal data to.
      \return True if the store was successful. False otherwise.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Called by the PowerLine durint stream deserialization. Restores internal
      data from the given stream.
      \param str - The StorageStream to restore internal data from.
      \param True if the restore was successful. False otherwise.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::Motor);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      virtual ~Motor() {}

      virtual agx::Real displacementToRatio(agx::Real displacement) override;
      virtual agx::Real constraintForceToPressure(agx::Real constraintForce, agx::Real displacement) const override;
      virtual agx::Real pressureToConstraintForce(agx::Real pressure, agx::Real displacement) const override;
  };



}

#endif
