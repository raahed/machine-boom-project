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
#include <agx/Plane.h>

namespace agx
{
  class Hinge;
}

namespace agxVehicle
{
  AGX_DECLARE_POINTER_TYPES( Wheel );

  /**
  Abstraction of a wheel of arbitrary geometry. The wheel has a well
  defined center, rotation axis and radius.
  */
  class AGXVEHICLE_EXPORT Wheel : public agx::Referenced, public agxStream::Serializable
  {
    public:
      using ConstraintContainer = agx::Vector<agx::ConstraintRef>;

    public:
      /**
      Construct wheel given radius, wheel body and frame defining wheel center,
      up and rotation axes in the rigid body frame.

      Rotation axis is by definition the y axis in \p rbRelFrame (if given, otherwise
      y axis of \p rb).

      Up axis is by definition the z axis in \p rbRelFrame (if given, otherwise z axis
      of \p rb).
      \param radius - radius of the wheel
      \param rb - wheel body
      \param rbRelFrame - relative frame defining center and rotation axis of this
                          wheel, given in \p rb frame.
      */
      Wheel( agx::Real radius, agx::RigidBody* rb, agx::Frame* rbRelFrame = nullptr );

      /**
      Construct wheel given radius, wheel body and transform defining wheel center,
      up and rotation axes in the rigid body frame.

      Rotation axis is by definition the y axis in \p rbRelFrame (if given, otherwise
      y axis of \p rb).

      Up axis is by definition the z axis in \p rbRelFrame (if given, otherwise z axis
      of \p rb).
      \param radius - radius of the wheel
      \param rb - wheel body
      \param rbRelTransform - relative transform defining center and rotation axis of this
                          wheel, given in \p rb frame.
      */
      Wheel( agx::Real radius, agx::RigidBody* rb, agx::AffineMatrix4x4 rbRelTransform );

      /**
      \return the radius of this wheel
      */
      agx::Real getRadius() const;

      /**
      \return the wheel rigid body
      */
      agx::RigidBody* getRigidBody() const;

      /**
      \return the local frame to the wheel body
      */
      agx::Frame* getLocalFrame() const;

      /**
      \return the transform of this wheel
      */
      agx::AffineMatrix4x4 getTransform() const;

      /**
      \return center position of this wheel, in world coordinate system
      */
      agx::Vec3 getCenterPosition() const;

      /**
      \return wheel axis in world coordinate system
      */
      agx::Vec3 getRotationAxis() const;

      /**
      \return wheel up axis in world coordinate system
      */
      agx::Vec3 getUpAxis() const;

      /**
      \return the plane defined by the rotation axis
      */
      agx::Plane getRotationAxisPlane() const;

      /**
      Create constraint with this wheel and some other rigid body. The constraint
      axis will point along the rotation axis and it's possible to offset the
      center along that same axis.

      The constraint is stored and accessible through the given name. The constraint
      will be added to the simulation when this wheel is used in an agxVehicle object.
      \param name - name of the constraint
      \param other - other rigid body
      \param wheelCenterOffset - offset along the rotation axis (from center of this wheel)
      \return constraint with this wheel body as first and \p other as second body in the constraint
      */
      template<typename T>
      agx::ref_ptr<T> attachConstraint( const agx::Name& name, agx::RigidBody* other, agx::Real wheelCenterOffset );

      /**
      Specialized version of attachConstraint to create a hinge with this wheel body and \p other.
      \sa attachConstraint
      \param name - name of the hinge
      \param other - other rigid body
      \param wheelCenterOffset - offset along the rotation axis (from center if this wheel)
      \return hinge with this wheel body as first and \p other as second body in the hinge
      */
      agx::Hinge* attachHinge( const agx::Name& name, agx::RigidBody* other, agx::Real wheelCenterOffset );

      /**
      \return vector with attached constraints created with attachConstraint
      */
      const ConstraintContainer& getConstraints() const;

      /**
      \return first constraint (null if no constraints has been attached with attachConstraint)
      */
      agx::Constraint* getConstraint() const;

      /**
      \param name - name of the constraint to find
      \return constraint with the given name
      */
      agx::Constraint* getConstraint( const agx::Name& name ) const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxVehicle::Wheel );

    protected:
      /**
      Default constructor used only when this object is restored.
      */
      Wheel();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~Wheel();

    private:
      void addConstraint( const agx::Name& name, agx::Constraint* constraint );

    private:
      agx::Real m_radius;
      agx::RigidBodyRef m_rb;
      agx::FrameRef m_rbRelFrame;
      ConstraintContainer m_constraints;
  };

  template<typename T>
  agx::ref_ptr<T> Wheel::attachConstraint( const agx::Name& name, agx::RigidBody* other, agx::Real wheelCenterOffset )
  {
    if ( m_rb == nullptr )
      return agx::ref_ptr<T>( nullptr );

    //           x            y               z
    const auto xAxis = getRotationAxis() ^ getUpAxis();
    agx::OrthoMatrix3x3 rotation( getUpAxis().x(), getUpAxis().y(), getUpAxis().z(),
                                  xAxis.x(), xAxis.y(), xAxis.z(),
                                  getRotationAxis().x(), getRotationAxis().y(), getRotationAxis().z() );

    const auto constraintFrame = agx::AffineMatrix4x4( rotation,
                                                       getCenterPosition() + wheelCenterOffset * getRotationAxis() );
    agx::FrameRef wheelFrame = new agx::Frame( constraintFrame * m_rb->getTransform().inverse() );
    agx::FrameRef otherFrame = new agx::Frame( constraintFrame * ( other != nullptr ? other->getTransform().inverse() : agx::AffineMatrix4x4() ) );

    agx::ref_ptr<T> constraint = new T( m_rb, wheelFrame, other, otherFrame );
    addConstraint( name, constraint );

    return constraint;
  }
}
