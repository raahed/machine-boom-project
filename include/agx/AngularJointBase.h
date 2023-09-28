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

#ifndef AGX_ANGULAR_JOINT_BASE_H
#define AGX_ANGULAR_JOINT_BASE_H

#include <agx/Constraint.h>

namespace agx
{
  typedef agx::Bool (*CreateElementaryConstraintsFunction)( class HighLevelConstraintImplementation* );

  /**
  An angular joint is defined by having 2 rotational degrees of freedom.
  The AngularJointBase is an abstract class.
  The other two rotational degrees of freedom can be controlled by controllers (motor, lock, range...)
  */
  class AGXPHYSICS_EXPORT AngularJointBase : public agx::Constraint
  {
    public:

      /**
      Create a angular joint given attachment frames and one or two rigid bodies. An
      attachment is frame relative its body with the joint axis point in the z-direction.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second attachment frame
      */
      AngularJointBase( CreateElementaryConstraintsFunction createFunc, agx::RigidBody* rb1, agx::Frame* rb1AttachmentFrame, agx::RigidBody* rb2 = nullptr, agx::Frame* rb2AttachmentFrame = nullptr  );

      /**
      \return the secondary constraint of type Lock1D for the given degree of freedom
      */
      agx::Lock1D*                          getLock1D( UInt dof );

      /**
      \return the secondary constraint of type Lock1D for the given degree of freedom
      */
      const agx::Lock1D*                    getLock1D( UInt dof ) const;

      /**
      \return the secondary constraint of type Motor1D for the given degree of freedom
      */
      agx::Motor1D*                         getMotor1D( UInt dof );

      /**
      \return the secondary constraint of type Motor1D for the given degree of freedom
      */
      const agx::Motor1D*                   getMotor1D( UInt dof ) const;

      /**
      \return the electric motor controller
      */
      agx::ElectricMotorController*         getElectricMotorController( UInt dof );

      /**
      \return the electric motor controller
      */
      const agx::ElectricMotorController*   getElectricMotorController( UInt dof ) const;

      /**
      \return the secondary constraint of type Range1D for the given degree of freedom
      */
      agx::Range1D*                         getRange1D( UInt dof );

      /**
      \return the secondary constraint of type Range1D for the given degree of freedom
      */
      const agx::Range1D*                   getRange1D( UInt dof ) const;

      /**
      \return the angle given degree of freedom
      */
      agx::Real                             getAngle( UInt dof ) const;

      /**
      \return the current speed given degree of freedom
      */
      agx::Real                             getCurrentSpeed( UInt dof ) const;

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override = 0;

      /**
      The constraint definition allows for controllers at all constraint angles.
      HOWEVER, for an angular joint, there is one angle not to be controlled, the first angle.
      When inheriting from the angular joint base an enum must define the controller indices
      for it to be possible to index the controllers.
      This enum will then have to start with its first value at the value of the number of non controlled angles

      One (1) is recommended as offset. The constraint inheriting defines its non controlled angle first,
      and then all the controlled. (Defining the angle is done by adding it to the attachment pair in the ::createFunc)

      \return the offset in the number of controllers
      */
      virtual UInt getNumControllerOffset() const = 0;

      /**
      \return the number of controller controlled degrees of freedom.
      */
      virtual int getNumControllers() const = 0;

      /**
      Returns angle not controlled by controllers
      */
      agx::Real getTheta() const;

      /**
      \returns first controller angle
      */
      agx::Real getPhi() const;

      /**
      \returns second controller angle
      */
      agx::Real getPsi() const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::AngularJointBase );

    protected:
      AngularJointBase();
      AngularJointBase(bool);
      virtual ~AngularJointBase();

      /**
      Used internally for debug rendering.
      */
      virtual void render(agxRender::RenderManager* mgr, float scale ) const override;

    protected:
      class AngularJointBaseImplementation* m_implementation;
  };

  typedef ref_ptr<AngularJointBase> AngularJointBaseRef;



} // namespace agx
#endif
