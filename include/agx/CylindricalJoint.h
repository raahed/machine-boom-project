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

#ifndef AGX_CYLINDRICAL_JOINT_H
#define AGX_CYLINDRICAL_JOINT_H

#include <agx/Constraint.h>

namespace agxUtil
{
  class Spline;
}

namespace agx
{
  /**
  Helper function to define the reference frame used by the prismatic constraint.
  */
  class CALLABLE AGXPHYSICS_EXPORT CylindricalJointFrame : public ConstraintFrame
  {
    public:
      /// Default constructor
      CylindricalJointFrame() {}

      // Constructor providing a point and an axis
      CylindricalJointFrame( const Vec3& point, const Vec3& axis );

      /**
      Set the translational axis.
      */
      void setAxis( const Vec3& axis );

      /**
      Provide a point lying on the axis
      */
      void setPoint( const Vec3& point );

      /**
      Configure everything at once by providing a line
      with given axis, passing through given point
      */
      void set(const Vec3 & point, const Vec3& axis );

      const Vec3& getAxis() const;
  };

  /**
  A cylindrical joint is similar to a prismatic joint but with an extra degree of
  freedom free (rotation around the axis).
  */
  class CALLABLE AGXPHYSICS_EXPORT CylindricalJoint : public Constraint2DOF
  {
    public:
      /**
      Create Cylindrical joint given a cylindrical joint frame (line defined in world coordinates)
      \param cf - Cylindrical joint frame in world coordinates
      \param rb1 - First rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      */
      CylindricalJoint( const CylindricalJointFrame& cf, RigidBody* rb1, RigidBody* rb2 = 0 );

      /**
      Create Cylindrical joint given one or two rigid bodies with corresponding attachment frames.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second rigid body attachment frame
      */
      CylindricalJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF
      {
        ALL_DOF=-1,        /**< Select all degrees of freedom */
        ROTATIONAL_1=0,    /**< Select DOF corresponding to the first rotational axis */
        ROTATIONAL_2=1,    /**< Select DOF corresponding to the second rotational axis */
        TRANSLATIONAL_1=2, /**< Select DOF for the first translational axis */
        TRANSLATIONAL_2=3, /**< Select DOF for the second translational axis */
        NUM_DOF=4          /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::CylindricalJoint);

    protected:
      CylindricalJoint();
      virtual ~CylindricalJoint();

      /**
      Internal method used for debug rendering
      */
      virtual void render( agxRender::RenderManager *mgr, float scale  ) const override;

    private:
      class CylindricalJointImplementation* m_implementation;
  };

  typedef ref_ptr<CylindricalJoint> CylindricalJointRef;

} // namespace agx
#endif
