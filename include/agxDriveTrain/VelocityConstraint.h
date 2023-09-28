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


#pragma once

#include <agx/Constraint.h>
#include <agx/ConstraintImplementation.h>
#include <agxPowerLine/RotationalDimension.h>


/*
This file contains various constraint related classes used by the FixedVelocityEngine.
*/


namespace agxDriveTrain
{
  DOXYGEN_START_INTERNAL_BLOCK()
  /**
  A velocity constraint that acts to keep the body moving in a fixed velocity.
  */
  class ElementaryVelocityConstraint : public agx::ElementaryConstraintN<1>
  {
    public:
      ElementaryVelocityConstraint();

      /**
      Writes 1 to the element index of G[row].
      */
      virtual agx::UInt getJacobian(
          agx::Jacobian6DOFElement* G, agx::UInt numBlocks,
          agx::UInt row, agx::GWriteState::Enum writeState) override;

      /**
      Writes the target velocity to v[row].
      */
      virtual agx::UInt getVelocity(agx::Real* v, agx::UInt row) const override;

      /**
      Writes 0 to v[row].
      */
      virtual agx::UInt getViolation(agx::Real* b, agx::UInt row) override;

      agx::Real getTargetVelocity();
      void setTargetVelocity(agx::Real targetVelocity);

      void setSlot(agx::UInt8 slot);

      virtual agx::Bool isImpacting() const override;

      virtual void storeLightData(agxStream::StorageStream& str) const override;
      virtual void restoreLightData(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::ElementaryVelocityConstraint);

    protected:
      virtual ~ElementaryVelocityConstraint() {}

    private:
      // To avoid warning C4512. And these really should not be copyable.
      void operator=(const ElementaryVelocityConstraint&) {}

    private:
      agx::Real m_targetVelocity;
      agx::UInt8 m_slot;
  };


  ElementaryVelocityConstraint* asElementaryVelocityConstraint(agx::Constraint* constraint);



  /**
  VelocityConstraintImplementation holding a single ElementaryVelocityConstraint.
  */
  class VelocityConstraintImplementation : public agx::ConstraintImplementation, public agxStream::Serializable
  {
    public:
      VelocityConstraintImplementation(agxPowerLine::RotationalUnit* rotationalUnit);
      void setRotationalDimension(agxPowerLine::RotationalDimension* rotationalDim);

      agx::Real getCurrentTorque();

      agx::Real getTargetVelocity();
      void setTargetVelocity(agx::Real targetVelocity);

      virtual bool updateValid() override;

      virtual void storeLightData(agxStream::StorageStream& str) const override;
      virtual void restoreLightData(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::VelocityConstraintImplementation);

    protected:
      virtual ~VelocityConstraintImplementation() {}
      VelocityConstraintImplementation();

    private:
      agxPowerLine::RotationalDimension* m_rotationalDimension;
      ElementaryVelocityConstraint* m_elementaryConstraint;
  };



  AGX_DECLARE_POINTER_TYPES(VelocityConstraint);

  /**
  A constraint holding a single VelocityConstraintImplementation.

  The VelocityConstraint tries to keep the body moving at a fixed velocity.
  */
  class VelocityConstraint : public agx::Constraint
  {
    public:
      /**
      Create a VelocityConstraint for the given rotational unit.
      */
      VelocityConstraint(agxPowerLine::RotationalUnit* rotationalUnit);
      void setRotationalDimension(agxPowerLine::RotationalDimension* rotationalDim);
      virtual void render(agxRender::RenderManager* canvas, float scale) const override;

      agx::Real getCurrentTorque();

      agx::Real getTargetVelocity();
      void setTargetVelocity(agx::Real targetVelocity);

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::VelocityConstraint);

    protected:
      VelocityConstraint();
      virtual ~VelocityConstraint();
      virtual int getNumDOF() const override;

    private:
      VelocityConstraintImplementation* m_velocityConstraint;
  };
  DOXYGEN_END_INTERNAL_BLOCK()

}
