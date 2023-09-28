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

#ifndef AGXCONTROL_ACTION_H
#define AGXCONTROL_ACTION_H

#include <agx/config.h>

#include <agx/Constraint.h>
#include <agx/RigidBody.h>
#include <agx/ParticleEmitter.h>
#include <agx/Referenced.h>
#include <agxStream/Serializable.h>

#include <agxControl/ActionImplementation.h>


namespace agxControl
{
  class EventSensor;

  class Operation;

  // These are the subclasses of Action that can be returned by the factory methods.
  AGX_DECLARE_POINTER_TYPES(ConstraintAction);
  AGX_DECLARE_POINTER_TYPES(SetEnableConstraintAction);
  AGX_DECLARE_POINTER_TYPES(SetComplianceConstraintAction);
  AGX_DECLARE_POINTER_TYPES(SetDampingConstraintAction);
  AGX_DECLARE_POINTER_TYPES(SetLockPositionToCurrentConstraintAction);

  AGX_DECLARE_POINTER_TYPES(SecondaryConstraintAction);
  AGX_DECLARE_POINTER_TYPES(BodyAction);
  AGX_DECLARE_POINTER_TYPES( ParticleEmitterAction );
  AGX_DECLARE_POINTER_TYPES( EmitterAction );
  AGX_DECLARE_POINTER_TYPES(EventSensorAction);


  AGX_DECLARE_POINTER_TYPES(Action);
  AGX_DECLARE_VECTOR_TYPES(Action);

  /**
  An Action represents some action that is to take place at some point in time.
  Actions are created using factory methods and are triggered during time stepping.
  The actual work is handled by instances of subclasses of the ActionImplementation
  class, which is a wrapper over std::function.

  The purpose of this higher level Action is to give a simple, controlled, serializable,
  and easily exported interface to a few well-defined actions.


  \internal

  The factory methods are very explicit. Can they be made more automatic? The signatures
  for a bunch of stuff was recently changed, r14800, and I had to manually change several
  places here. Worse, the change required additions to the agxStream::Storage<T> lists,
  omissions that didn't appear until runtime, and a few dynamic_casts, which also wasn't
  discovered until runtime. We're gonna need a lot of unit tests for this.

  May want to move the time aspect out of Action since actions may be triggered
  by many things. For example, the ActionManager could be renamed to TimedActionManager
  and keep the time for each event. The add(Event*) method could be changed to
  schedule(Event*, Real time) instead.


  Experimenting with many classes. See section below. However, SpaceClaim doesn't
  need the inspection capabilities at the moment so I'll leave it as it is. Later,
  when we know more about what Actions we need, we may be able to find a middle
  ground where Actions that are similar, such as setCompliance and setDamping, are
  the same class.



  The current implementation has only a few subclasses of Action based on the
  type of object being manipulated. More detailed information, such as the method
  to call and the parameters to pass, are hidden inside an ActionImplementation
  instance. This approach makes it hard to inspect and manipulate already created
  Actions. There is no easy way of knowing what an Action will do. The only way
  is a large if-else chain of string comparisons and templated getValue#? calls,
  or a if-else chain of dynamic_casts on the return value from getImplementation.

  The alternative would be another level of subclassing, creating a setCompliance
  action that inherits from ConstraintAction. Would that even help? What do we
  want to do with the Actions? Display in a GUI I assume. Change the values use
  when the event is triggered as well.
  */
  class AGXPHYSICS_EXPORT Action : public agx::Referenced, public agxStream::Serializable
  {
  public:
    /* A bunch of static factory methods. */

    /// Create an Action that enables the given constraint.
    static SetEnableConstraintAction* setEnableConstraint(agx::Real time, agx::Constraint* constraint, bool enable);

    /**
    Create an Action that sets the compliance of a constraint. The 'dof'
    argument is the degree of freedom that should be modified and should be
    a member of the DOF enum for the type of constraint that the 'constraint'
    parameter points to.
    */
    static SetComplianceConstraintAction* setCompliance(agx::Real time, agx::Constraint* constraint, agx::Real newCompliance, agx::Int dof);
    static SetDampingConstraintAction* setDamping(agx::Real time, agx::Constraint* constraint, agx::Real newDamping, agx::Int dof);
    static SecondaryConstraintAction* enableMotor(agx::Real time, agx::Constraint1DOF* constraint, bool enable);
    static SecondaryConstraintAction* enableMotor(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, bool enable);
    static SecondaryConstraintAction* setMotorSpeed(agx::Real time, agx::Constraint1DOF* constraint, agx::Real speed);
    static SecondaryConstraintAction* setMotorSpeed(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real speed);
    static SecondaryConstraintAction* setMotorMinForceRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real force);
    static SecondaryConstraintAction* setMotorMinForceRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real force);
    static SecondaryConstraintAction* setMotorMaxForceRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real force);
    static SecondaryConstraintAction* setMotorMaxForceRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real force);
    static SecondaryConstraintAction* setMotorCompliance(agx::Real time, agx::Constraint1DOF* constraint, agx::Real compliance);
    static SecondaryConstraintAction* setMotorCompliance(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real compliance);
    static SecondaryConstraintAction* setMotorDamping(agx::Real time, agx::Constraint1DOF* constraint, agx::Real damping);
    static SecondaryConstraintAction* setMotorDamping(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real damping);
    static SecondaryConstraintAction* enableRange(agx::Real time, agx::Constraint1DOF* constraint, bool enable);
    static SecondaryConstraintAction* enableRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, bool enable);
    static SecondaryConstraintAction* setRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real lower, agx::Real upper);
    static SecondaryConstraintAction* setRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real lower, agx::Real upper);
    static SecondaryConstraintAction* setRangeMinRange(agx::Real, agx::Constraint1DOF* constraint, agx::Real limit);
    static SecondaryConstraintAction* setRangeMinRange(agx::Real, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real limit);
    static SecondaryConstraintAction* setRangeMaxRange(agx::Real, agx::Constraint1DOF* constraint, agx::Real limit);
    static SecondaryConstraintAction* setRangeMaxRange(agx::Real, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real limit);
    static SecondaryConstraintAction* setRangeMinForceRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real force);
    static SecondaryConstraintAction* setRangeMinForceRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real force);
    static SecondaryConstraintAction* setRangeMaxForceRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real force);
    static SecondaryConstraintAction* setRangeMaxForceRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real force);
    static SecondaryConstraintAction* setRangeCompliance(agx::Real time, agx::Constraint1DOF* constraint, agx::Real compliance);
    static SecondaryConstraintAction* setRangeCompliance(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real compliance);
    static SecondaryConstraintAction* setRangeDamping(agx::Real time, agx::Constraint1DOF* constraint, agx::Real damping);
    static SecondaryConstraintAction* setRangeDamping(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real damping);
    static SecondaryConstraintAction* enableLock(agx::Real time, agx::Constraint1DOF* constraint, bool enable);
    static SecondaryConstraintAction* enableLock(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, bool enable);
    static SecondaryConstraintAction* setLockMinForceRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real force);
    static SecondaryConstraintAction* setLockMinForceRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real force);
    static SecondaryConstraintAction* setLockMaxForceRange(agx::Real time, agx::Constraint1DOF* constraint, agx::Real force);
    static SecondaryConstraintAction* setLockMaxForceRange(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real force);
    static SecondaryConstraintAction* setLockPosition(agx::Real time, agx::Constraint1DOF* constraint, agx::Real position);
    static SecondaryConstraintAction* setLockPosition(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real position);
    static SecondaryConstraintAction* setLockCompliance(agx::Real time, agx::Constraint1DOF* constraint, agx::Real compliance);
    static SecondaryConstraintAction* setLockCompliance(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real compliance);
    static SecondaryConstraintAction* setLockDamping(agx::Real time, agx::Constraint1DOF* constraint, agx::Real damping);
    static SecondaryConstraintAction* setLockDamping(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real damping);

    static SecondaryConstraintAction* setLockPositionToCurrent(agx::Real time, agx::Constraint1DOF* constraint);
    static SecondaryConstraintAction* setLockPositionToCurrent(agx::Real time, agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

    static BodyAction* setEnableBody(agx::Real time, agx::RigidBody* body, bool enable);
    static BodyAction* setBodyVelocity(agx::Real time, agx::RigidBody* body, agx::Vec3 velocity);

    static ParticleEmitterAction* setEnableParticleEmitter( agx::Real time, agx::ParticleEmitter* emitter, bool enable );
    static ParticleEmitterAction* setParticleEmitterQuantityFlowRate( agx::Real time, agx::ParticleEmitter* emitter, agx::Real flowrate );
    static EmitterAction* setEnableEmitter( agx::Real time, agx::Emitter* emitter, bool enable );
    static EmitterAction* setEmitterQuantityFlowRate( agx::Real time, agx::Emitter* emitter, agx::Real flowrate );
    static EventSensorAction* setEnableEventSensor(agx::Real time, agxControl::EventSensor* sensor, bool enable);

    /// \return The time when the action should be triggered.
    agx::Real getTime() const;

    /**
    Trigger the Action, causing whatever it action represents to happen.
    \param currentTime The current time of the simulation. This may be different
                       from the time returned by getTime() because of the discrete
                       nature of the time stepping.
    */
    void trigger(agx::Real currentTime);

    /// Equivalent to action->trigger(action->getTime());
    void trigger();

    /// \return The ActionImplementation that holds the callback to run when this Action is triggered.
    ActionImplementation* getImplementation();
    const ActionImplementation* getImplementation() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::Action );


  protected:
    /// Used only for store/restore.
    Action();


    Action(const agx::String& target, const agx::String& operation, ActionImplementation* implementation);

    const agx::String& getTargetType() const;
    const agx::String& getOperationType() const;

    friend class agxControl::Operation;

    /**
    Swap the 'm_implementation' member between this Action and the given one.
    No validity checks are performed, so use with care.
    */
    void swapImplementation(Action* other);


    // Each Action target type, the classes that an Action can manipulate, has a
    // string ID used during store/restore operations.
    static const agx::String& getConstraintActionId();
    static const agx::String& getSecondaryConstraintActionId();
    static const agx::String& getSetPositionFromCurrentConstraintActionId();
    static const agx::String& getBodyActionId();
    static const agx::String& getGeometryActionId();
    static const agx::String& getVoidActionId();
    static const agx::String& getParticleEmitterActionId();
    static const agx::String& getEmitterActionId();
    static const agx::String& getEventSensorActionId();

    // Each operation that an Action can perform has a string ID used during store/
    // restore operations.
    static const agx::String& getEnableOperationId();
    static const agx::String& getFlowRateOperationId();
    static const agx::String& getComplianceOperationId();
    static const agx::String& getDampingOperationId();
    static const agx::String& getSpeedOperationId();

    static const agx::String& getRangeOperationId();
    static const agx::String& getRangeMinOperationId();
    static const agx::String& getRangeMaxOperationId();

    static const agx::String& getForceRangeMinOperationId();
    static const agx::String& getForceRangeMaxOperationId();

    static const agx::String& getPositionOperationId();
    static const agx::String& getVelocityOperationId();

    static const agx::String& getVoidOperationId();

  protected:
    virtual ~Action();


  protected:
    ActionImplementationRef m_implementation;

  private:
    agx::String m_targetType;
    agx::String m_operationType;
  };


  /**
  An Action that operates on constraints. Instances of this class should only be
  created only through the Action factory methods.
  */
  class AGXPHYSICS_EXPORT ConstraintAction : public agxControl::Action
  {
  public:
    /// \return The constraint that this Action will manipulate.
    agx::Constraint* getConstraint();

  public:
    /// For use only by the serialization framework.
    ConstraintAction() {}
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::ConstraintAction );

  protected:
    virtual ~ConstraintAction();

    friend class Action;
    ConstraintAction(agx::Constraint* constraint, const agx::String& operation, ActionImplementation* implementation);

  private:
    agx::ConstraintRef m_constraint;
  };


  /**
  An Action that sets the compliance of a constraint. The sole purpose of this
  class is to provide an interface to inspect and manipulate this particular
  type of event.
  */
  class AGXPHYSICS_EXPORT SetComplianceConstraintAction : public agxControl::ConstraintAction
  {
  public:
    agx::Real getCompliance() const;
    void setCompliance(agx::Real newCompliance);
    void setCompliance(agx::Real newCompliance, int dof);

    Callback2ActionImplementation<agx::Real, int>* getTypedImplementation();
    const Callback2ActionImplementation<agx::Real, int>* getTypedImplementation() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::SetComplianceConstraintAction );
    SetComplianceConstraintAction() {}

  protected:
    virtual ~SetComplianceConstraintAction() {}

  private:
    friend class Action;
    SetComplianceConstraintAction(agx::Constraint* constraint, const agx::String& operation, ActionImplementation* implementation);
  };



  /**
  An Action that sets the damping of a constraint. The sole purpose of this
  class is to provide an interface to inspect and manipulate this particular
  type of event.
  */
  class AGXPHYSICS_EXPORT SetDampingConstraintAction : public agxControl::ConstraintAction
  {
  public:
    agx::Real getDamping() const;
    void setDamping(agx::Real newDamping);
    void setDamping(agx::Real newDamping, int dof);

    Callback2ActionImplementation<agx::Real, int>* getTypedImplementation();
    const Callback2ActionImplementation<agx::Real, int>* getTypedImplementation() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::SetDampingConstraintAction );
    SetDampingConstraintAction() {}

  protected:
    virtual ~SetDampingConstraintAction() {}

  private:
    friend class Action;
    SetDampingConstraintAction(agx::Constraint* constraint, const agx::String& operation, ActionImplementation* implementation);
  };


  /**
  An Action that enables or disables a constraint. The sole purpose of this
  class is to provide an interface to inspect and manipulate this particular
  type of event.
  */
  class AGXPHYSICS_EXPORT SetEnableConstraintAction : public agxControl::ConstraintAction
  {
  public:
    bool getEnabling() const;
    void setEnabling(bool enable);

    Callback1ActionImplementation<bool>* getTypedImplementation();
    const Callback1ActionImplementation<bool>* getTypedImplementation() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::SetEnableConstraintAction );
    SetEnableConstraintAction() {}

  protected:
    virtual ~SetEnableConstraintAction();

  private:
    friend class Action;
    SetEnableConstraintAction(agx::Constraint* constraint, const agx::String& operation, ActionImplementation* implementation);
  };



  /**
  An Action that operates on the secondary constraints of a regular constraints.
  It is used, for example, to enable or disable motors, locks, and ranges, set
  motor speeds, or range ranges.

  Instances of this class should only be created only through the Action factory methods.
  */
  class AGXPHYSICS_EXPORT SecondaryConstraintAction : public agxControl::Action
  {
  public:
    /// \return The constraint that this Action will manipulate.
    agx::Constraint* getConstraint();

  public:
    /// For use only by the serialization framework.
    SecondaryConstraintAction() {}
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::SecondaryConstraintAction );

  protected:
    ~SecondaryConstraintAction();

    friend class Action;
    SecondaryConstraintAction(agx::Constraint* constraint, const agx::String& operation, const agx::String& secondaryTarget, ActionImplementation* implementation);

    // The list of secondary constraint targets that a SecondaryConstraintAction can operate on.
    static const agx::String& getMotorId();
    static const agx::String& getMotor1Id();
    static const agx::String& getMotor2Id();
    /// \return getMotor1Id for DOF::FIRST and getMotor2Id for DOF::SECOND. Is getErrorId if anything goes wrong.
    static const agx::String& getMotorId(agx::Constraint2DOF::DOF dof);
    static const agx::String& getRangeId();
    static const agx::String& getRange1Id();
    static const agx::String& getRange2Id();
    /// \return getRange1Id for DOF::FIRST and getRange2Id for DOF::SECOND. Is getErrorId if anything goes wrong.
    static const agx::String& getRangeId(agx::Constraint2DOF::DOF dof);
    static const agx::String& getLockId();
    static const agx::String& getLock1Id();
    static const agx::String& getLock2Id();
    /// \return getLock1Id for DOF::FIRST and getLock2Id for DOF::SECOND. Is getErrorId if anything goes wrong.
    static const agx::String& getLockId(agx::Constraint2DOF::DOF dof);
    static const agx::String& getErrorId();

    /// \return A string identifying the secondary constraint that this action will manipulate.
    const agx::String& getSecondaryTarget() const;

    void restoreEnableActionImplementation();
    void restoreSpeedOperation();
    void restoreRangeOperation();
    void restoreRangeMinOperation();
    void restoreRangeMaxOperation();
    void restoreForceRangeMinOperation();
    void restoreForceRangeMaxOperation();
    void restoreComplianceOperation();
    void restoreDampingOperation();
    void restorePositionOperation();

  protected:
    agx::ConstraintRef m_constraint;

  private:
    agx::String m_secondaryTarget;
  };


  /**
  An Action that operates on a RigidBody. Instances of this class should only be
  created only through the Action factory methods.
  */
  class AGXPHYSICS_EXPORT BodyAction : public agxControl::Action
  {
  public:
    /// \return The body that this Action will manipulate.
    agx::RigidBody* getRigidBody();

  public:
    /// For use by the serialization framework only.
    BodyAction();
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::BodyAction );

  protected:
    friend class Action;
    BodyAction(agx::RigidBody* body, const agx::String& operation, ActionImplementation* implementation);

  private:
    agx::RigidBodyRef m_body;
  };

  /**
  An Action that operates on an ParticleEmitter. Instances of this class should only be
  created only through the Action factory methods.
  */
  class AGXPHYSICS_EXPORT ParticleEmitterAction : public agxControl::Action
  {
  public:
    /// \return The body that this Action will manipulate.
    agx::ParticleEmitter* getParticleEmitter();

  public:
    /// For use by the serialization framework only.
    ParticleEmitterAction();
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::ParticleEmitterAction );

  protected:
    friend class Action;
    ParticleEmitterAction(agx::ParticleEmitter* emitter, const agx::String& operation, ActionImplementation* implementation);

  private:
    agx::ParticleEmitterRef m_emitter;
  };

  /**
  An Action that operates on an Emitter. Instances of this class should only be
  created only through the Action factory methods.
  */
  class AGXPHYSICS_EXPORT EmitterAction : public agxControl::Action
  {
  public:
    /// \return The body that this Action will manipulate.
    agx::Emitter* getEmitter();

  public:
    /// For use by the serialization framework only.
    EmitterAction();
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::EmitterAction );

  protected:
    friend class Action;
    EmitterAction( agx::Emitter* emitter, const agx::String& operation, ActionImplementation* implementation );

  private:
    agx::EmitterRef m_emitter;
  };

  /**
  An Action that operates on an EventSensor. Instances of this class should only be
  created only through the Action factory methods.
  */
  class AGXPHYSICS_EXPORT EventSensorAction : public agxControl::Action
  {
  public:
    /// \return The body that this Action will manipulate.
    EventSensor* getEventSensor();

  public:
    /// For use by the serialization framework only.
    EventSensorAction();
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::EventSensorAction );

  protected:
    friend class Action;
    EventSensorAction(EventSensor* sensor, const agx::String& operation, ActionImplementation* implementation);

  private:
    EventSensor * m_sensor;
  };

  /**
  Custom action that can be created from any ActionImplementation. Used to create
  actions that do things not supported by the factory methods.

  Cannot be serialized.
  */
  class AGXPHYSICS_EXPORT CallbackAction : public agxControl::Action
  {
  public:
    CallbackAction(ActionImplementation* implementation);

  public:
    /// For use by the serialization framework only.
    CallbackAction() {}
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::CallbackAction );
  };

  class AGXPHYSICS_EXPORT SetLockPositionToCurrentConstraintAction : public agxControl::SecondaryConstraintAction
  {
  public:
    CallbackActionImplementation* getTypedImplementation();
    const CallbackActionImplementation* getTypedImplementation() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::SetLockPositionToCurrentConstraintAction );
    SetLockPositionToCurrentConstraintAction();

    void executeAction(agx::Real time);
  protected:
    virtual ~SetLockPositionToCurrentConstraintAction();

  private:
    friend class Action;
    SetLockPositionToCurrentConstraintAction(agx::Constraint1DOF* constraint, agx::Real time);
    SetLockPositionToCurrentConstraintAction(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof, agx::Real time);

    agx::Constraint2DOF::DOF m_dof;
  };
}

// Include guard
#endif

/// \endcond
