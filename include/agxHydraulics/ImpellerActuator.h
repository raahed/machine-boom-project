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


#ifndef AGXHYDRAULICS_IMPELLER_ACTUATOR_H
#define AGXHYDRAULICS_IMPELLER_ACTUATOR_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/Pipe.h>
#include <agxHydraulics/RotationalFlowConnector.h>
#include <agxPowerLine/Actuator1DOF.h>
#include <agxPowerLine/ActuatorConnector.h>
#include <agxPowerLine/ActuatorUnit.h>

namespace agx
{
  class RigidBody;
}

namespace agxModel
{
  class Connector;
}

namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(ImpellerActuator);

  /**
  Base class for Actuators that convert between hydraulic flow and mechanical
  6-DOF rotation, e.g., the HydraulicPumpActuator and the
  HydraulicMotorActuator. Contains, in addition to the regular Actuator
  elements, a FlowUnit that represents the fluid chamber and a Connector that
  connects the fluid chamber to the rotational dimension of the mechanics
  bodies.

  Connecting a FlowUnit to the ImpellerActuator, or connecting the
  ImpellerActuator to a FlowUnit, will connect the FlowUnit to the
  ImpellerActuator's internal chamber.
  */
  class AGXHYDRAULICS_EXPORT ImpellerActuator : public agxPowerLine::RotationalActuator
  {
    public:
      /**
      Set the relationship between hydraulic flow and mechanical rotation.
      Measured in volume of fluid per radian of mechanical rotation.
      */
      void setDisplacement(agx::Real displacement);

      /**
      \return The displacement of the ImpellerActuator.
      */
      agx::Real getDisplacement() const;

      /**
      \return The FlowUnit that represent the fluid chamber of the actuator.
      */
      agxHydraulics::FlowUnit* getChamber();
      const agxHydraulics::FlowUnit* getChamber() const;


      /**
      \return
      */
      agxHydraulics::RotationalFlowConnector* getRotationalFlowConnector();
      const agxHydraulics::RotationalFlowConnector* getRotationalFlowConnector() const;

      virtual void setEnable(bool enabled) override;
      virtual bool getEnable() const override;

      // Methods used by the rest of the framework.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
#ifndef SWIG
      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
          agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side) override;

      virtual void getConnectableDimensionTypes(
          agxPowerLine::PhysicalDimension::TypeVector& types, agxPowerLine::Side side) const override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;


      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxHydraulics::ImpellerActuator);

      void restore(agxStream::InputArchive& in) override;

      void store(agxStream::OutputArchive& out) const override;
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Create an ImpellerActuator that joins the fluid chamber with the two
      bodies attached with the given hinge. Will produce rotations about the
      hinge axis.
      */
      ImpellerActuator(agx::Hinge* hinge, agx::Real chamberLength, agx::Real chamberArea, agx::Real chamberDensity);
      ImpellerActuator();

      virtual ~ImpellerActuator();

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Create the Connector that connects the fluid chamber to the 6-DoF bodies.
      Different subclasses creates different types Connectors. The created
      Connector is stored in the m_impellerChamberConnector member variable.
      */
      virtual void createRotationalFlowConnector() = 0;


      virtual bool internalConnect(
        agxPowerLine::ActuatorBodyUnit* flowSideBody,
        agxPowerLine::ActuatorBodyUnit* oppositeSideBody,
        agxHydraulics::FlowUnit* flowUnit,
        agxPowerLine::Side connectorFlowSide);
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      FlowUnitRef                m_flowUnit;
      RotationalFlowConnectorRef m_impellerChamberConnector;
  };







  AGX_DECLARE_POINTER_TYPES(HydraulicMotorActuator);

  /**
  An Actuator that converts hydraulic flow into rotation of a hinge.
  */
  class AGXHYDRAULICS_EXPORT HydraulicMotorActuator : public agxHydraulics::ImpellerActuator
  {
    public:
      /**
      Creates a motor actuator that causes rotation about the given hinge when
      there is flow in the internal chamber.

      \param hinge - The hinge that should be powered.
      \param chamberLength - The length of the internal fluid chamber.
      \param chamberArea - The area of the internal fluid chamber.
      \param fluidDensity - The density of the fluid inside the internal chamber.
      */
      HydraulicMotorActuator(agx::Hinge* hinge, agx::Real chamberLength, agx::Real chamberArea, agx::Real fluidDensity);


      /**
      \return The Connector that connects the motor chamber to the mechanical 6-DoF bodies.
      */
      agxHydraulics::RotationalFlowConnector* getMotorConnector();
      const agxHydraulics::RotationalFlowConnector* getMotorConnector() const;

    // Methods called by the rest of the PowerLine/Hydraulics framework.
    public:
#ifndef SWIG
      /**
      \internal
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      \internal
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::HydraulicMotorActuator);
#endif

  protected:
    HydraulicMotorActuator();
    virtual ~HydraulicMotorActuator();

    virtual void createRotationalFlowConnector() override;
  };






  AGX_DECLARE_POINTER_TYPES(HydraulicPumpActuator);

  /**
  An Actuator that converts rotation of a hinge into hydraulic flow.
  */
  class AGXHYDRAULICS_EXPORT HydraulicPumpActuator : public agxHydraulics::ImpellerActuator
  {
  public:
    /**
    Creates a pump actuator that causes flow in the internal chamber when there
    is rotation about the given hinge.

    \param hinge - The hinge that should power the fluid.
    \param chamberLength - The length of the internal fluid chamber.
    \param chamberArea - The area of the internal fluid chamber.
    \param chamberDensity - The density of the fulid inside the internal chamber.
    */
    HydraulicPumpActuator(agx::Hinge* hinge, agx::Real chamberLength, agx::Real chamberArea, agx::Real chamberDensity);

    /**
    \return The Connector that connets the mechanical 6-DoF bodies to the pump chamber.
    */
    agxHydraulics::RotationalFlowConnector* getPumpConnector();
    const agxHydraulics::RotationalFlowConnector* getPumpConnector() const;


#ifndef SWIG
    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;
    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::HydraulicPumpActuator);
#endif

  protected:
    HydraulicPumpActuator();
    virtual ~HydraulicPumpActuator();

    virtual void createRotationalFlowConnector() override;
  };



}

#endif
