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

#ifndef AGX_PRISMATIC_UNIVERSAL_JOINT_H
#define AGX_PRISMATIC_UNIVERSAL_JOINT_H

#include <agx/AngularJointBase.h>

namespace agx
{
  /**
  A Prismatic Universal Joint has two translational dof's constrained.
  And one constrained axis (the rotation around the N axis of each frame the bodies)
  */
  class AGXPHYSICS_EXPORT PrismaticUniversalJoint : public agx::AngularJointBase
  {
    public:

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF {
        ALL_DOF = -1,      /**< Select all degrees of freedom */
        ROTATIONAL_1 = 0,  /**< An angular joint will always have one rotational degree of freedom controlled */
        TRANSLATIONAL_1 = 1, /**< Select DOF for the first translational axis */
        TRANSLATIONAL_2 = 2, /**< Select DOF for the second translational axis */
        NUM_DOF         = 3
      };

      /**
      Enum to access the controllers for unconstrained DOF's of the constraint

      Since the universal joint inherits from the AngularJointBase, NUM_CONTROLLER_OFFSET is defined to be one.
      To loop over all controllers, start at NUM_CONTROLLER_OFFSET and loop until <= NUM_CONTROLLER_DOF.
      */
      enum CONTROLLER_DOF {
        NUM_CONTROLLER_OFFSET   = 1,
        ROTATIONAL_CONTROLLER_1 = 0 + NUM_CONTROLLER_OFFSET, /** Select first controller of the prismatic universal joint, control angle*/
        ROTATIONAL_CONTROLLER_2 = 1 + NUM_CONTROLLER_OFFSET, /** Select second controller of the prismatic universal joint, control angle*/
        TRANSLATIONAL_CONTROLLER_1 = 2 + NUM_CONTROLLER_OFFSET,/** Select third controller of the prismatic universal joint, control separation*/
        NUM_CONTROLLER_DOF      = 3

      };

      /**
      Create a prismatic universal joint given attachment frames and one or two rigid bodies. An
      attachment is frame relative its body with the joint axis point in the z-direction.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second attachment frame
      */
      PrismaticUniversalJoint( agx::RigidBody* rb1, agx::Frame* rb1AttachmentFrame, agx::RigidBody* rb2 = nullptr, agx::Frame* rb2AttachmentFrame = nullptr );

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      /**
      Constraints in AGX allow for controllers for all non constraint angles.
      HOWEVER, for an angular joint, there is one angle not to be controlled, the first angle.
      When inheriting from the angular joint base an enum must define the controller indices
      for it to be possible to index the controllers.
      This enum will then have to start with its first value at the value of the number of non controlled angles

      One (1) is recommended as offset. The constraint inheriting defines its non controlled angle first,
      and then all the controlled. (Defining the angle is done by adding it to the attachment pair in the ::createFunc)

      \return the offset in the number of controllers
      */
      virtual UInt getNumControllerOffset() const override;

      /**
      \return the number of controller controlled degrees of freedom.
      */
      virtual int getNumControllers() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::PrismaticUniversalJoint );

    protected:
      PrismaticUniversalJoint();
      virtual ~PrismaticUniversalJoint();

  };

  typedef ref_ptr<PrismaticUniversalJoint> PrismaticUniversalJointRef;



} // namespace agx
#endif
