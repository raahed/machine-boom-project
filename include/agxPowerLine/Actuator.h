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


#ifndef AGXPOWERLINE_ACTUATOR_H
#define AGXPOWERLINE_ACTUATOR_H

#include <agxPowerLine/SubGraph.h>
#include <agxPowerLine/ActuatorUnit.h>
#include <agxPowerLine/Sides.h>
#include <agxPowerLine/Unit.h>

namespace agx
{
  class RigidBody;
}

namespace agxPowerLine
{
  class PhysicalDimension;
  class Connector;

  namespace detail
  {
    class AbstractDimensionState3Dof;
  }
}

namespace agxPowerLine
{
  AGX_DECLARE_POINTER_TYPES(Actuator);

  /**
   *
   */
  class AGXMODEL_EXPORT Actuator : public agxPowerLine::Unit
  {
    public:

      Actuator( agx::Constraint* constraint );

      const agx::Constraint* getConstraint() const;
      agx::Constraint* getConstraint();

      virtual agx::Real calculateRelativeGradient() const = 0;
      virtual agx::Real calculateRelativeValue() const = 0;

      virtual agx::Vec3 calculateLocalDirection(Side side, int dimensionType) const = 0;
      virtual agx::Vec3 calculateWorldDirection(Side side, agxPowerLine::PhysicalDimension::Type dimensionType) const = 0;

      virtual const agx::RigidBody* getRigidBody(Side side) const;
      virtual agx::RigidBody* getRigidBody(Side side);

      agxPowerLine::PhysicalDimension* getPhysicalDimension(Side side, int dimensionType);
      const agxPowerLine::PhysicalDimension* getPhysicalDimension(Side side, int dimensionType) const;

      agxPowerLine::detail::Translational3DofState* getTranslationalState(Side side);
      agxPowerLine::detail::Rotational3DofState* getRotationalState(Side side);

      agxPowerLine::detail::AbstractDimensionState3Dof* getDimensionState(
          agxPowerLine::Side side,
          agxPowerLine::PhysicalDimension::Type dimensionType);

      bool reverseOrderRelativeConstraint() const;

      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
          agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side) override;


      agxPowerLine::ActuatorBodyUnit* getActuatorBodyUnit(Side side);
      const agxPowerLine::ActuatorBodyUnit* getActuatorBodyUnit(Side side) const;

      //* Vectors in world coordinates to calculate the Jacobian */
      virtual agx::Vec3 getDir1( ) const = 0;
      virtual agx::Vec3 getDir2( ) const = 0;
      virtual agx::Vec3 getCmToAnchorPos1() const = 0;
      virtual agx::Vec3 getCmToAnchorPos2() const = 0;
      virtual agx::Vec3 getSeparation() const = 0;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      using agxPowerLine::SubGraph::store;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      using agxPowerLine::SubGraph::restore;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxMode::Actuator);
      virtual void store(agxStream::OutputArchive& out) const override;
      virtual void restore(agxStream::InputArchive& in) override;

    protected:
      virtual ~Actuator();

      Actuator();

    protected:
      bool m_enabled;

      agx::observer_ptr<agx::Constraint> m_constraint;

      //agx::observer_ptr<agxPowerLine::ActuatorBodyUnit> m_actuatorBodyUnits[Actuator::NUM_BODIES];
      agx::ref_ptr<agxPowerLine::ActuatorBodyUnit> m_actuatorBodyUnits[2];
  };


}

#endif
