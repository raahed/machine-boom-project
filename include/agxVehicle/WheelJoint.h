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

#include <agxVehicle/export.h>
#include <agx/Constraint.h>


namespace agxVehicle
{

  AGX_DECLARE_POINTER_TYPES(WheelJoint);


  /**
  Helper class to define reference frames for WheelJoint
  */
  class CALLABLE AGXVEHICLE_EXPORT WheelJointFrame : public agx::ConstraintFrame
  {
    public:

      /**
      The WheelJoint has a direction for the wheel axle, a direction for the steering axle
      which must be orthogonal to the wheel axle and an intersection point for the two axes.

      The two input axes will be normalized. The steering axle will be adjusted if needed
      so that it is orthogonal to the wheel axle.
      \param point Intersection point for wheel axle and steering axle in world coordinates
      \param wheelAxle World direction for wheel axle
      \param steeringAxle World direction for steering axle
      */
      WheelJointFrame( agx::Vec3 point, agx::Vec3 wheelAxle, agx::Vec3 steeringAxle );

  };



  /**
  The wheel constraint is designed to attach two bodies such that one of
  them is free to rotate about an axis defined in its own frame, the wheel axle,
  and about an axis defined in the other body, which would be the
  chassis. This second axis becomes the steering axis and is also the
  direction for suspension.

  The wheel axis and steering axis must be perpendicular.
  \todo {
  This could be relaxed to provide for camber, toe, and caster angles. Stay
  posted! }

  This allows for rotation about the center of the wheel around the axle as
  well as rotation about the steering axis. The wheel is also allowed to move
  up/down along the steering axis to have suspension.

  By default, there is no control on steering or driving. Suspension is initially
  locked.  By using the DOFController object and setting flags and other parameters,
  the relative angles can be computed and the extension for suspension
  can be controlled.
  */
  class CALLABLE AGXVEHICLE_EXPORT WheelJoint : public agx::Constraint {
    public:

      /**
      Construct a WheelJoint from a WheelJointFrame and bodies.
      \param wjf A WheelJointFrame specifying directions and point
      \param rb1 Rigid body for the wheel.
      \param rb2 Rigid body for the chassi. Passing a nullptr for rb2 is allowed, then world is used.
      */
      WheelJoint( const WheelJointFrame& wjf, agx::RigidBody* rb1, agx::RigidBody* rb2 );

      /**
      Constructor to create a wheel from two rigid bodies, a center point and a steering/suspension axis.
      \param rb1 - First rigid body (invalid if null). Wheel axle will be along this body's Y-axis.
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param center - intersection point of steering axis and wheel rotation axis given in world coordinates
      \param axis - steering axis given in world coordinates
      \
      */
      WheelJoint(agx::RigidBody* rb1, agx::RigidBody* rb2, const agx::Vec3& center, const agx::Vec3& axis);


      /**
      Create wheel given body attachment frames and one or two rigid bodies.
      The wheel axle is defined to be along the Y-axis for the first attachment frame.
      The steering axle and suspension is along the Z-axis for body 2.
      If rb2 == rb2AttachmentFrame == nullptr, then first body will be attached to the world given
      rb1AttachmentFrame and in this case, the world z-axis will be the direction of the suspension.
      If rb2 == nullptr and rb2AttachmentFrame != nullptr, then rb2AttachmentFrame will be used as the world frame attachment.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second rigid body attachment frame (invalid if rb2 != 0 and this is null)
      */
      WheelJoint(agx::RigidBody* rb1, agx::Frame* rb1AttachmentFrame, agx::RigidBody* rb2 = nullptr,
                 agx::Frame* rb2AttachmentFrame = nullptr);


      /**
      Get the rigid body of the wheel
      \return the wheel's rigid body
      */
      agx::RigidBody* getWheelRigidBody();

      /**
      Get the rigid body of the chassis. If the wheel is attached to the world, a nullptr will be returned.
      \return the chassis' rigid body
      */
      agx::RigidBody* getChassisRigidBody();

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF {
        ALL_DOF = -1,        /**< Select all degrees of freedom */
        TRANSLATIONAL_1 = 0, /**< Select DOF for the first translational axis (U) in relation to suspension axis. */
        TRANSLATIONAL_2 = 1, /**< Select DOF for the second translational axis (V) in relation to suspension axis. */
        ROTATIONAL_1 = 3,    /**< Select DOF corresponding to the first rotational axis. Alignment between wheel axis and steering axis. */
        NUM_DOF = 3          /**< Number of DOF available for this constraint */
      };

      /**
      Enum for specifying which secondary constraint to access
      */
      enum SecondaryConstraint {
        STEERING = 0,        /**< For control of the steering axis */
        WHEEL = 1,           /**< For control of wheel rotation */
        SUSPENSION = 2,      /**< For control of the suspension */

        STEERINGCONSTRAINT_BOUNDS = 3,  /**< Extra bounds that are available for a steering constraint
                                             to use (e.g. an AckermannSteering constraint). This secondary
                                             constraint should not be used by the API user. Instead use STEERING. */
        NUM_FREEDOF = 3
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints.
      */
      int getNumDOF() const override;

      /**
      \return the secondary constraint of type Motor1D for secondary constraint sc
      */
      agx::Motor1D* getMotor1D(SecondaryConstraint sc);

      /**
      \return the secondary constraint of type Lock1D for secondary constraint sc
      */
      agx::Lock1D* getLock1D(SecondaryConstraint sc);

      /**
      \return the secondary constraint of type Range1D for secondary constraint sc
      */
      agx::Range1D* getRange1D(SecondaryConstraint sc);


      /**
      \return The angle for the given degree of freedom
      */
      agx::Real getAngle( SecondaryConstraint sc = STEERING ) const;

      /**
      Perform debug rendering of this constraint via agxRender::RenderManager
      */
      void render(class agxRender::RenderManager* mgr, float scale) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxVehicle::WheelJoint);

      WheelJoint();

    protected:
      virtual ~WheelJoint();

    private:
      class WheelJointImplementation* m_implementation;
  };

} //namespace agx
