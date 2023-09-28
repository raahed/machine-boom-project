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

namespace agxUtil
{
  class Spline;
}

namespace agx
{

  /**
  A SplineJoint is a constraint with a motor/range/lock for translation/rotation that will restrict a body to move along a spline path.
  \note This class is a bit experimental.
        During interpolation, the closest point of the curve will be evaluated. This might cause a jump from the original curve segment to a nearby segment.
  */
  class AGXPHYSICS_EXPORT SplineJoint : public Constraint2DOF
  {
  public:

    /**
    Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
    constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
    */
    enum DOF
    {
      ALL_DOF = -1,        /**< Select all degrees of freedom */
      ROTATIONAL_1 = 0,    /**< Select DOF corresponding to the first rotational axis */
      ROTATIONAL_2 = 1,    /**< Select DOF corresponding to the second rotational axis */
      TRANSLATIONAL_1 = 2, /**< Select DOF for the first translational axis */
      TRANSLATIONAL_2 = 3, /**< Select DOF for the second translational axis */
      NUM_DOF = 4          /**< Number of DOF available for this constraint */
    };


    /**
    Create SplineJoint given one rigid body with corresponding attachment frame.
    \param rb - Rigid body (invalid if null)
    \param rbAttachmentFrame - Rigid body attachment frame (invalid if null)
    \param spline - Spline that define a curve onto which the body will be attached.
    */
    SplineJoint(RigidBody* rb, Frame* rbAttachmentFrame, agxUtil::Spline* spline);

    /**
    \return the number of DOF for this constraint, not including secondary constraints. -1 if
    num DOF is undefined for the constraint.
    */
    virtual int getNumDOF() const override;
    virtual void render(class agxRender::RenderManager * mgr, float scale = 1.0f) const override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::SplineJoint);

  protected:

    // For restore only
    SplineJoint();

    virtual ~SplineJoint();

  private:
    class SplineJointImplementation* m_implementation;
  };

  AGX_DECLARE_POINTER_TYPES(SplineJoint);


} // namespace agx
