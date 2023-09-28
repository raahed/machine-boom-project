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


#ifndef AGXPOWERLINE_ACTUATOR_1DOF_H
#define AGXPOWERLINE_ACTUATOR_1DOF_H

#include <agxPowerLine/Actuator.h>
#include <agxPowerLine/TranslationalDimension.h>
#include <agxPowerLine/TranslationalUnit.h>
#include <agxPowerLine/RotationalUnit.h>
#include <agxPowerLine/detail/RotationalActuatorConnector.h>
#include <agx/Prismatic.h>
#include <agx/DistanceJoint.h>
#include <agx/Hinge.h>
#include <agxPowerLine/detail/TranslationalActuatorConnector.h>

namespace agx
{
  class RigidBody;
}

namespace agxPowerLine
{
  AGX_DECLARE_POINTER_TYPES(Actuator1DOF);

  class AGXMODEL_EXPORT Actuator1DOF : public agxPowerLine::Actuator
  {
    public:

      Actuator1DOF(agx::Constraint1DOF* constraint1DOF);

      // Begin inhertied from agxPowerLine::Actuator.
      virtual agx::Real calculateRelativeValue() const override;
      // End inhertied from PowerLine::Actuator.

      // Begin inhertied from PowerLine::Unit.
      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
          agxPowerLine::PhysicalDimension::Type type,
          agxPowerLine::Side side) override;
      // End inherited from PowerLine::Unit.


      // Begin inherited from PowerLine::SubGraph.
      virtual bool preUpdate(agx::Real timeStamp) override;
      // End inherited from PowerLine::SubGraph

      void synchronizeDirections();

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      using agxPowerLine::Actuator::store;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      using agxPowerLine::Actuator::restore;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::Actuator1DOF);
      void store(agxStream::OutputArchive& out) const override;
      void restore(agxStream::InputArchive& in) override;

    protected:
      Actuator1DOF();

      virtual ~Actuator1DOF();

      void synchronizeUnits( agx::Constraint1DOF* constraint );
      void replaceActuatorUnit( Unit* oldUnit, Unit* newUnit );

      void detachActuatorUnitsFromEachOther();

      virtual agx::Vec3 getDir1() const override;
      virtual agx::Vec3 getDir2() const override;
      virtual agx::Vec3 getCmToAnchorPos1() const override;
      virtual agx::Vec3 getCmToAnchorPos2() const override;
      virtual agx::Vec3 getSeparation() const override;


    protected:
      /**
      Observer that not only sets the Constraint1DOF pointer to nullptr on object
      deletion, but also informs the owning Actuator so the body units can be updated.
      */
      class ConstraintObserver : public agx::observer_ptr<agx::Constraint1DOF>
      {
        public:
          ConstraintObserver(agx::Constraint1DOF* actuatedConstraint, Actuator1DOF* owningActuator);
          virtual void objectDeleted(void*) override;

          using agx::observer_ptr<agx::Constraint1DOF>::operator=;
        private:
          Actuator1DOF* m_owningActuator;
      };

      friend class ConstraintObserver;

    protected:
      ConstraintObserver m_constraint1DOF;
      agx::ref_ptr<agxPowerLine::ActuatorBodyUnit> m_dummyActuatorBodyUnits[2];
  };



  class AGXMODEL_EXPORT RotationalActuator : public agxPowerLine::Actuator1DOF
  {
    public:
      RotationalActuator( agx::Hinge* hinge );

      /**
      \returns true if high speed mode is active
      */
      agx::Bool getUseHighSpeedMode( ) const;

      /**
      Set the usage of high speed mode.

      RotationalActuator has a feature enabling for high hinge velocities.
      When enabled the relative angular velocity orthogonal to the hinge axis
      will be the same for the two rigid bodies being hinged.

      This results in the possibility to have hinges with arbitrary high velocities,
      which is needed when using hinges as high speed engines/motors.
      */
      void setUseHighSpeedMode( agx::Bool highSpeedMode );

      /**
      \return The shaft that other power line components may connect to. Rotation
              in this shaft is transfered to the Hinge to which this Rotational-
              Actuator is connected.
      */
      agxPowerLine::RotationalUnit* getInputShaft();


      virtual void setEnable(bool enabled);
      virtual bool getEnable() const;

    // Called by the rest of the power line framework.
    public:
      virtual bool preUpdate(agx::Real timeStamp) override;

      virtual void getConnectableDimensionTypes(
        agxPowerLine::PhysicalDimension::TypeVector& types,
        agxPowerLine::Side side) const override;

      virtual agx::Real calculateRelativeGradient() const override;

      virtual agx::Vec3 calculateWorldDirection(
          Side side,
          agxPowerLine::PhysicalDimension::Type dimensionType) const override;

      virtual agx::Vec3 calculateLocalDirection(
          Side side,
          int dimensionType) const override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& out) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& in) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::RotationalActuator);

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
          agxPowerLine::PhysicalDimension::Type type,
          agxPowerLine::Side side) override;


    protected:
      RotationalActuator();

      /*
      There are subclasses of RotationalActuator that want to create their own
      versions of the input shaft and the internal connector. For example, the
      hydraulics package contains the ImpellerActuator, which uses a FlowUnit
      for input.

      Any subclass that uses the dont_create_input tag when constructing the
      RotationalActuator base class must implement get- and setEnable.

      This part of the inheritence tree could use some redesign. There should
      be one base class, inheriting from Actuator1DOF, that knows about hinges
      and how to get Jacobians from them, and a collection of subclasses
      derived of off that that knows about the type of the input and the custom
      constraint that binds the input PhysicalDimension to the
      ActuatorBodyUnits.

      Perhaps we should even move the concept of input unit and internal
      constraint up a bit more and put it inside Actuator1DOF. In that case we
      don't have to deal with the oddities of the current setEnable
      implementation, which is just bad. A base class implementation does the
      wrong thing for the derived classes.

      Not sure if the name RotationalActuator should be used for the base class
      or the subclass that has a RotationalUnit input. The latter is more
      consistent with the current implementation and a better user-level name
      (the base class is not user-level), but what should the base class be
      named?
      */
      struct AGXMODEL_EXPORT dont_create_input_t {};
      static dont_create_input_t dont_create_input;
      RotationalActuator(agx::Hinge* hinge, dont_create_input_t);

      virtual ~RotationalActuator();

      void connectActuatorConnector();
      void disconnectActuatorConnector();
      void reconnectActuatorConnector();
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      agxPowerLine::RotationalUnitRef m_inputShaft;
      agxPowerLine::detail::RotationalActuatorConnectorRef m_actuatorConnector;

      agx::Bool m_useHighSpeedMode;
  };

  typedef agx::ref_ptr<RotationalActuator> RotationalActuatorRef;



  class AGXMODEL_EXPORT TranslationalActuator : public agxPowerLine::Actuator1DOF
  {
    public:
      TranslationalActuator(agx::DistanceJoint* distance);

      TranslationalActuator(agx::Prismatic* prismatic);

      agxPowerLine::TranslationalUnit* getInputRod();


    // Called by the rest of the power line framework.
    public:
      virtual void getConnectableDimensionTypes(
          agxPowerLine::PhysicalDimension::TypeVector& types,
          agxPowerLine::Side side) const override;

      /**
      Stores internal data into stream.
      */
      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
            agxPowerLine::PhysicalDimension::Type type,
            agxPowerLine::Side side) override;

      /**
      Restores internal data from stream.
      */
      virtual agx::Real calculateRelativeGradient() const override;

      virtual agx::Vec3 calculateWorldDirection(
          Side side,
          agxPowerLine::PhysicalDimension::Type dimensionType) const override;

      virtual agx::Vec3 calculateLocalDirection(Side side, int dimensionType) const override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::TranslationalActuator);

    protected:
      TranslationalActuator();

      struct AGXMODEL_EXPORT dont_create_input_t {};
      static dont_create_input_t dont_create_input;

      TranslationalActuator(agx::DistanceJoint* distance, dont_create_input_t);
      TranslationalActuator(agx::Prismatic* prismatic, dont_create_input_t);

      virtual ~TranslationalActuator();

      void initialize();

      void connectActuatorConnector();

    private:
      agxPowerLine::TranslationalUnitRef m_inputRod;
      agxPowerLine::detail::TranslationalActuatorConnectorRef m_actuatorConnector;
  };

  typedef agx::ref_ptr<TranslationalActuator> TranslationalActuatorRef;


}

#endif
