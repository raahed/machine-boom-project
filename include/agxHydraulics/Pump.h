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


#ifndef AGXHYDRAULICS_PUMP_H
#define AGXHYDRAULICS_PUMP_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/RotationalFlowConnector.h>

namespace agxHydraulics
{

  AGX_DECLARE_POINTER_TYPES(Pump);

  /**
  A PowerLine Connector that connects to a rotational input and a flow output.
  Rotation at the input is converted to flow at the output according to the
  configured displacement. The displacement is measured in terms of volume per
  unit of rotation, measured in radians. For simulations using the SI unit
  system, a pump displacement of 1.0 will move 1.0 cubic meters of fluid per
  radian of rotation at the input, or 2*pi cubic meter per revolution.
  Displacements are often much smaller than this.
  */
  class AGXHYDRAULICS_EXPORT Pump : public agxHydraulics::RotationalFlowConnector
  {
    public:
      /**
      Creates a new Pump with displacement 1.
      */
      Pump();

      /**
      Create a new Pump with the specified displacement.
      \param displacement - The rotation-to-flow conversion factor.
      */
      explicit Pump(agx::Real displacement);


    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Called by the PowerLine during initialization.
      */
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

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

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::Pump);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()


    protected:
      virtual ~Pump() {}

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Used by RotationalFlowConnector when preparing for Jacobian calculation.
      */
      virtual agx::Real displacementToRatio(agx::Real displacement) override;

      /**
      Used by the RotationalFlowConnector.
      */
      virtual agx::Real constraintForceToPressure(agx::Real constraintForce, agx::Real displacement) const override;

      /**
      Used by the RotationalFlowConnector.
      */
      virtual agx::Real pressureToConstraintForce(agx::Real pressure, agx::Real displacement) const override;
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      void init();
  };

}

#endif
