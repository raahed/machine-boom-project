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



/// \cond CONTROL

#ifndef AGXCONTROL_DYNAMIC_OPERATION_H
#define AGXCONTROL_DYNAMIC_OPERATION_H


#include <agx/config.h>

#include <agxControl/Operation.h>
#include <agxControl/InterpolationController.h>

#include <agx/Constraint.h>

namespace agxControl
{

  AGX_DECLARE_POINTER_TYPES(EndOperationAction);

  /**
  An Action whose sole purpose is to mark the end of an dynamic operation.
  */
  class AGXPHYSICS_EXPORT EndOperationAction : public agxControl::Action
  {
  public:
    EndOperationAction(agx::Real time);
    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::EndOperationAction);
    EndOperationAction(){};
  protected:
    virtual ~EndOperationAction() {}
  };


  class ConstraintAngleOperation;

  /**
  Base class for the operations that set action parameters based on the current
  state of the simulation.

  /// \internal

  This class is currently empty. Is it even needed?
  Was intended to provide factory methds for the various DynamicOperations we
  have, but the very explicit sub-classes made it redundant. May be needed if we
  decide to do templated DynamicOperations to reduce code duplication.
  */
  class AGXPHYSICS_EXPORT DynamicOperation : public agxControl::Operation
  {
  protected:
    virtual ~DynamicOperation() {}
  };


  /**
  DynamicOperation that manipulates a Constraint1DOF motor's speed so that it
  reaches a given angle at the end of the operation time interval.
  */
  class AGXPHYSICS_EXPORT ConstraintAngleOperation : public agxControl::Operation
  {
  public:
    ConstraintAngleOperation(
      agx::Real startTime, agx::Real endTime, agx::Constraint1DOF* constraint, agx::Real targetAngle, bool useHermite = true);

    void setWinding(agx::Real winding);
    agx::Real getWinding() const;

    virtual void activate(agx::Real time) override;
    virtual void update(agx::Real time) override;
    virtual void deactivate(agx::Real time) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::ConstraintAngleOperation );
    ConstraintAngleOperation() {}
  protected:
    virtual ~ConstraintAngleOperation() {}

  private:
    agxControl::SecondaryConstraintAction* getWorkAction();
    agxControl::Callback1ActionImplementation<agx::Real>* getWorkActionImplementation();
    agx::Constraint1DOF* getConstraint();

  private:
    bool m_useHermite;

    // Interpolation data.
    agx::Real m_beginAngle; // Read during activate.
    agx::Real m_endAngle;   // Supplied by user through constructor.
    agx::Real m_beginSpeed; // Read during activate.
    agx::Real m_endSpeed;   // May be set by user, initially zero.

    agx::Real m_winding; // Constraint winding interval. May be set by user, initially zero.

    agxControl::HermiteInterpolationControllerRef m_hermite;
  };


  class AGXPHYSICS_EXPORT BodyPositionOperation : public agxControl::Operation
  {
  public:
    BodyPositionOperation(agx::Real startTime, agx::Real endTime, agx::RigidBody* body, agx::Vec3 targetPosition);

    virtual void activate(agx::Real time) override;
    virtual void update(agx::Real time) override;
    virtual void deactivate(agx::Real time) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::BodyPositionOperation );
    BodyPositionOperation() {}
  protected:
    virtual ~BodyPositionOperation() {}

  private:
    agxControl::BodyAction* getWorkAction();
    agxControl::Callback1ActionImplementation<agx::Vec3>* getWorkImplementation();
    agx::RigidBody* getBody();

  private:
    agx::Vec3 m_targetPosition;
  };
}

// Include guard.
#endif
/// \endcond CONTROL
