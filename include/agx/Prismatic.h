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

#ifndef AGX_PRISMATIC_H
#define AGX_PRISMATIC_H

#include <agx/Constraint.h>

namespace agx
{
  /**
  Helper function to define the reference frame used by the prismatic constraint.
  */
  class CALLABLE AGXPHYSICS_EXPORT PrismaticFrame : public ConstraintFrame
  {
    public:
      /**
      Default constructor: translational axis is along Z.
      */
      PrismaticFrame();

      /// Destructor
      ~PrismaticFrame();

      /**
      Constructor
      \param point - a point lying on the axis
      \param axis - translational axis
      */
      PrismaticFrame( const Vec3& point, const Vec3& axis );

      /** Set the translational axis. */
      void setAxis( const Vec3& axis );

      /** Provide a point lying on the axis */
      void setPoint( const Vec3& point );

      /** Configure everything at
          once by providing a line
          with given axis, passing
          through given point
      */
      void set(const Vec3& point, const Vec3& axis );

      /// \return the translational axis
      const Vec3& getAxis() const;
  };

  /**
  A prismatic joint keeps a fixed relative orientation between the
  attached bodies but allows them to slide along a given axis.
  Hydraulics and elevators are examples of using a Prismatic constraint.
  */
  class CALLABLE AGXPHYSICS_EXPORT Prismatic : public Constraint1DOF
  {
    public:
      /**
      Create a Prismatic joint given a prismatic frame (world coordinates) and one or
      two rigid bodies.
      \param pf - Prismatic frame, defining the prismatic axis and attach point
      \param rb1 - First rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      */
      Prismatic( const PrismaticFrame& pf, RigidBody* rb1, RigidBody* rb2 = 0 );

      /**
      Create a Prismatic joint given attachment frames and one or two rigid bodies. An
      attachment is frame relative its body with the prismatic axis point in the z-direction.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second attachment frame
      */
      Prismatic( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      /**
      Create a Prismatic joint given the prismatic axis and one or two rigid bodies.
      \param axis - Prismatic axis given in world coordinates
      \param rb1 - Fist rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      */
      Prismatic( const Vec3& axis, RigidBody* rb1, RigidBody* rb2 = nullptr );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF {
        ALL_DOF = -1,      /**< Select all degrees of freedom */
        ROTATIONAL_1 = 0,  /**< Select DOF corresponding to the first rotational axis */
        ROTATIONAL_2 = 1,  /**< Select DOF corresponding to the second rotational axis */
        ROTATIONAL_3 = 2,  /**< Select DOF for rotation around Z-axis */
        TRANSLATIONAL_1 = 3, /**< Select DOF for the first translational axis */
        TRANSLATIONAL_2 = 4, /**< Select DOF for the second translational axis */
        NUM_DOF = 5        /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Prismatic );

    protected:
      Prismatic();
      virtual ~Prismatic();

      /**
      Used internally for debug rendering.
      */
      virtual void render(agxRender::RenderManager* mgr, float scale ) const override;

    private:
      class PrismaticImplementation* m_implementation;
  };

  typedef ref_ptr<Prismatic> PrismaticRef;



} // namespace agx
#endif
