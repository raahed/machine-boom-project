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

#ifndef AGXPOWERLINE_ACTUATOR_UNIT_H
#define AGXPOWERLINE_ACTUATOR_UNIT_H

#include <agxPowerLine/RotationalDimension.h>
#include <agxPowerLine/TranslationalDimension.h>
#include <agxPowerLine/detail/DimensionState.h>
#include <agxPowerLine/Unit.h>

namespace agx
{
  class RigidBody;
  class Constraint;
}

namespace agxPowerLine
{
  class Actuator;
  class RotationalUnit;
  class ActuatorBodyUnit;
}

namespace agxPowerLine
{
  class ActuatorDimension
  {
    public:
      ActuatorDimension(ActuatorBodyUnit* actuatorBodyUnit);

      void setActuatorBodyUnit(ActuatorBodyUnit* actuatorBodyUnit);
      const ActuatorBodyUnit* getActuatorBodyUnit() const;

      virtual agxPowerLine::detail::AbstractDimensionState3Dof* getDimensionState() = 0;
      virtual agx::RigidBody* getBody() = 0;

      bool store(agxStream::StorageStream& str) const;
      bool restore(agxStream::StorageStream& str);

      void store(agxStream::OutputArchive& out) const;
      void restore(agxStream::InputArchive& in);

    protected:
      ActuatorDimension();
      virtual ~ActuatorDimension() = 0;

    protected:
      agx::observer_ptr<ActuatorBodyUnit> m_actuatorBodyUnit;
  };




  class ActuatorTranslationalDimension : public agxPowerLine::TranslationalDimension, public ActuatorDimension
  {
    public:
      ActuatorTranslationalDimension(ActuatorBodyUnit* actuatorBodyUnit, agx::RigidBody* body, bool externalBody);

      virtual agxPowerLine::detail::AbstractDimensionState3Dof* getDimensionState() override;
      agxPowerLine::detail::Translational3DofState* getTranslationalDimensionState();

      virtual agx::RigidBody* getBody() override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::ActuatorTranslationalDimension);

    protected:
      virtual ~ActuatorTranslationalDimension();

    private:
      ActuatorTranslationalDimension();

    private:
      agxPowerLine::detail::Translational3DofState* m_dimensionState;
  };




  class ActuatorRotationalDimension : public agxPowerLine::RotationalDimension, public ActuatorDimension
  {
    public:
      ActuatorRotationalDimension(ActuatorBodyUnit* actuatorBodyUnit, agx::RigidBody* body, bool externalBody);

      virtual agxPowerLine::detail::AbstractDimensionState3Dof* getDimensionState() override;
      agxPowerLine::detail::Rotational3DofState* getRotationalDimensionState();

      agx::RigidBody* getBody() override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::ActuatorRotationalDimension);

    protected:
      virtual ~ActuatorRotationalDimension();

    private:
      ActuatorRotationalDimension();

    private:
      agxPowerLine::detail::Rotational3DofState* m_dimensionState;
  };




  /**
  An ActuatorBodyUnit forms a connection between the power line world and the
  rest of the simulation. It contains rotational- and translational actuator
  dimensions that bind to a regular \p agx::RigidBody.
  */
  class AGXMODEL_EXPORT ActuatorBodyUnit : public agxPowerLine::Unit
  {
  public:
    /**
    Create a rotational and translational unit.
    */
    ActuatorBodyUnit(Actuator* actuator, agx::Constraint* constraint, agxPowerLine::Side actuatorSide);

    const Actuator* getActuator() const;
    Actuator* getActuator();

    bool isInputAndOutput() const;

    const agx::RigidBody* getRigidBody() const;

    /**
    returns a pointer to the rotational dimension
    */
    agxPowerLine::ActuatorRotationalDimension* getRotationalDimension();

    /**
    returns a pointer to the rotational dimension
    */
    agxPowerLine::ActuatorTranslationalDimension* getTranslationalDimension();


    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::ActuatorBodyUnit);

    ActuatorBodyUnit();

  protected:

    virtual ~ActuatorBodyUnit();

    agx::ref_ptr<ActuatorRotationalDimension> m_rotationalDimension;
    agx::ref_ptr<ActuatorTranslationalDimension> m_translationalDimension;
    agx::observer_ptr<Actuator> m_actuator;
  };

  typedef agx::ref_ptr<ActuatorBodyUnit> ActuatorBodyUnitRef;

}

#endif // AGXMODEL_ACTUATOR_UNIT_H
