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

#include <agx/agx.h>
#include <agx/RigidBody.h>
#include <agx/Frame.h>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4355) //Disable 'this' : used in base member initializer list warning
#endif

namespace agx
{
  class DynamicsSystem;
  class AutoSleep;
  class MergeSplitBodyStorage;

  AGX_DECLARE_POINTER_TYPES( ObserverFrame );
  AGX_DECLARE_VECTOR_TYPES( ObserverFrame );
  typedef agx::SetVector<ObserverFrameRef> ObserverFrameRefSetVector;


  /**
  With this class you can attach an ObserverFrame object relative to a RigidBody. It allows for queering position/rotation/velocity in
  a coordinate system that is local to this ObserverFrame.
  */
  class CALLABLE AGXPHYSICS_EXPORT ObserverFrame : public agx::Referenced, public virtual agxStream::Serializable
  {
    public:


    public:
      /**
      Construct given name. Default: rigid body = nullptr/world, transform = identity
      \param name - name of this observer frame (default: "")
      \param rb - the rigid body to attach to, by default world
      \param transform - the frame of reference of the ObserverFrame
      */
      ObserverFrame( const agx::Name& name = "", agx::RigidBody* rb = nullptr, const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4());
      /**
      Construct without name. Default: transform = identity
      \param rb - the rigid body to attach to, null if world
      \param transform - the frame of reference of the ObserverFrame
      */
      ObserverFrame(agx::RigidBody* rb , const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4());

      /**
      Returns the model frame containing model the transformation and
      utilities to manipulate position, rotation etc.
      \return model frame of this observer frame
      */
      agx::Frame* getFrame();

      /**
      Returns the model frame containing model the transformation and
      utilities to manipulate position, rotation etc.
      \return model frame of this observer frame
      */
      const agx::Frame* getFrame() const;

      /**
      This method will set the local matrix of this frame to be the matrix \p m
      multiplied with the inverse of the parents matrix:
        M = m * inv(parent)
      I.e., assign the final (world) transform of this frame.
      \param matrix - final transform for this frame (leads to local transform = matrix * inv(parent))
      */
      void setTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return the local transformation matrix (i.e., the transformation relative to the parent)
      */
      const agx::AffineMatrix4x4& getLocalTransform() const;

      /**
      Assign the local transformation matrix for this frame ignoring any eventual parent transformation.
      \param matrix - relative to parent transform for this frame
      */
      void setLocalTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return the relative translate to this frame's parent frame
      */
      agx::Vec3 getLocalPosition() const;

      /**
      Assign the parent relative translate of this frame.
      \param p - parent relative translate
      */
      void setLocalPosition( const agx::Vec3& p );

      /**
      Assign the parent relative translate of this frame.
      \param x - parent relative x translate
      \param y - parent relative y translate
      \param z - parent relative z translate
      */
      void setLocalPosition( agx::Real x, agx::Real y, agx::Real z );

      /**
      Set the rotation of the frame relative to world frame.
      \param q - rotation given as a quaternion
      */
      void setRotation( const agx::Quat& q );

      /**
      Set the rotation of the frame relative to world frame.
      \param e - rotation given as Euler angles
      */
      void setRotation( const agx::EulerAngles& e);

      /**
      Set the rotation of the frame relative to world frame.
      \param m - rotation given as an orthogonal transformation matrix
      */
      void setRotation( const agx::OrthoMatrix3x3& m );

      /**
      \return the relative rotation to this frame's parent frame
      */
      agx::Quat getLocalRotation() const;

      /**
      Assign the parent relative rotation of this frame.
      \param q - rotation given as a quaternion
      */
      void setLocalRotation( const agx::Quat& q );

      /**
      Assign the parent relative rotation of this frame.
      \param e - rotation given as Euler angles
      */
      void setLocalRotation( const agx::EulerAngles& e );

      /**
      \return a pointer to the parent frame, 0 if this frame has no parent
      */
      agx::Frame* getParentFrame();

      /**
      \return a pointer to the parent frame, 0 if this frame has no parent
      */
      const agx::Frame* getParentFrame() const;

      /**
      Set the parent Frame of this ObserverFrame.

      This means that getTranslate, getRotate, getTransform will be given
      in the parents coordinate frame. I.e., this frame's transformation
      will be concatenated with the parents.
      \param frame - new parent frame, 0 to remove parent
      \return true if parent is changed (not the same as before) - otherwise false
      */
      bool setParentFrame( agx::Frame* frame );

      /**
      Assign the final (world) translate of this frame.
      \param p - final (world) translate of this frame
      */
      void setPosition( const agx::Vec3& p );

      /**
      Assign the final (world) translate of this frame.
      \param x - final (world) x translate of this frame
      \param y - final (world) y translate of this frame
      \param z - final (world) z translate of this frame
      */
      void setPosition( agx::Real x, agx::Real y, agx::Real z );

      /**
      Change state enable of this observer frame.
      Default: true.
      \param enable - true to enable, false to disable
      */
      void setEnable( bool enable );

      /**
      Access the state enable flag.
      \return true if the body is enabled (default) - otherwise false
      */
      bool getEnable() const;

      /**
      Access the state enable flag.
      \return true if the body is enabled (default) - otherwise false
      */
      bool isEnabled() const;

      /**
      \return a pointer to the DynamicsSystem this observer frame belongs to
      */
      agx::DynamicsSystem* getSystem();

      /**
      \return a pointer to the DynamicsSystem this observer frame belongs to
      */
      const agx::DynamicsSystem* getSystem() const;


      /**
      Current model frame position, given in world coordinate frame.
      \return the current model frame position, given in world coordinate frame
      */
      agx::Vec3 getPosition() const;

      /**
      Current model frame rotation, given in world coordinate frame.
      \return the current model frame rotation, given in world coordinate frame
      */
      agx::Quat getRotation() const;

      /**
      Wrapper functions for Frame
      */
      agx::Vec3 transformVectorToWorld(const agx::Vec3& vectorLocal) const;
      agx::Vec3 transformVectorToLocal(const agx::Vec3& vectorWorld) const;
      agx::Vec3 transformPointToWorld(const agx::Vec3& pointLocal) const;
      agx::Vec3 transformPointToLocal(const agx::Vec3& pointWorld) const;

      /**
        Get the velocity of the ObserverFrame in the world
      */
      agx::Vec3 getVelocity() const;

      /**
      Get the angular velocity of the ObserverFrame in the world
      */
      agx::Vec3 getAngularVelocity() const;

      /**
      Get the acceleration of the ObserverFrame in the world
      */
      agx::Vec3 getAcceleration() const;

      /**
      Get the angular acceleration of the ObserverFrame in the world
      */
      agx::Vec3 getAngularAcceleration() const;

      /**
      Get the relative velocity of a rigid body and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeVelocity(const agx::RigidBody* rb) const;

      /**
      Get the relative angular velocity of a rigid body and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeAngularVelocity(const agx::RigidBody* rb) const;

      /**
      Get the relative acceleration of a rigid body and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeAcceleration(const agx::RigidBody* rb) const;

      /**
      Get the relative angular acceleration of a rigid body and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeAngularAcceleration(const agx::RigidBody* rb) const;

      /**
      Get the relative velocity of an observer frame and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeVelocity(const agx::ObserverFrame* obs) const;

      /**
      Get the relative angular velocity of an observer frame and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeAngularVelocity(const agx::ObserverFrame* obs) const;

      /**
      Get the relative acceleration of an observer frame and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeAcceleration(const agx::ObserverFrame* obs) const;

      /**
      Get the relative angular acceleration of an observer frame and the ObserverFrame in the world
      */
      agx::Vec3 getRelativeAngularAcceleration(const agx::ObserverFrame* obs) const;

      /**
      Get the relative position (distance) of an observer frame and the ObserverFrame in the world
      */
      agx::Vec3 getRelativePosition(const agx::ObserverFrame* obs) const;

      /**
      Get the relative position (distance) of an observer frame and the RigidBody in the world
      */
      agx::Vec3 getRelativePosition(const agx::RigidBody* body) const;

      /**
      Attach the ObserverFrame to a rigid body. This will remove previous attachments.
      \arg rb The rigid body to attach to, null if world
      \arg transform The transform relative to the body
      */
      void attach(agx::RigidBody* rb, const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4());

      /**
      Attach the ObserverFrame to a rigid body. This will remove previous attachments.
      \arg rb The rigid body to attach to, null if world
      \arg transform The transform in the world
      */
      void attachWithWorldTransform(agx::RigidBody* rb, const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4());

      /**
      Detach the current rigid body, setting the current rigid body to the world and the frame to identity
      */
      void detach();

      agx::RigidBody* getAttachment() const;

      /**
      \return the name of this observer frame
      */
      agx::Name getName() const;

      /**
      \return the rigid body this observer frame is attached to
      */
      const agx::RigidBody* getRigidBody() const;

      /**
      \return the rigid body this observer frame is attached to
      */
      agx::RigidBody* getRigidBody();

      /**
      Assign new name to this observer frame.
      Default: ""
      */
      void setName( const agx::Name& name );

      /**
      Set the specified inertia in the frame of the observer on the rigid body.
      Also set the center of mass to the origin of the observer
      \param rb The rigid body to change the inertia on
      \param inertia The inertia in the frame of the observer
      \return False if fails. This means that an invalid inertia is used. Otherwise True
      */
      bool setCmAndInertiaInObserver(agx::RigidBody* rb, const agx::SPDMatrix3x3& inertia);

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::ObserverFrame );

    protected:
      friend class DynamicsSystem;

      virtual ~ObserverFrame();

      /**
      Assign dynamics system which this body belongs to.
      \param system - dynamics system which this body belongs to
      */
      void setSystem( agx::DynamicsSystem* system );

    private:
      void init(const agx::Name& name, agx::RigidBody* rb, const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4() );

    private:
      DynamicsSystem* m_mainSystem;

      FrameRef m_frame;
      bool m_enabled;
      agx::Name m_name;
      agx::RigidBodyRef m_rigidBody;
  };

  AGX_FORCE_INLINE bool ObserverFrame::getEnable() const
  {
    return m_enabled;
  }

  AGX_FORCE_INLINE bool ObserverFrame::isEnabled() const
  {
    return m_enabled;
  }

  AGX_FORCE_INLINE Frame* ObserverFrame::getFrame()
  {
    return m_frame.get();
  }

  AGX_FORCE_INLINE const Frame* ObserverFrame::getFrame() const
  {
    return m_frame.get();
  }

  inline const agx::RigidBody* ObserverFrame::getRigidBody() const
  {
    return m_rigidBody.get();
  }

  inline  agx::RigidBody* ObserverFrame::getRigidBody()
  {
    return m_rigidBody.get();
  }

} // namespace agx

#ifdef _MSC_VER
# pragma warning(pop)
#endif

