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

#ifndef AGX_RIGIDBODY_H
#define AGX_RIGIDBODY_H

#include <agx/agx.h>
#include <agx/Frame.h>
#include <agx/MassProperties.h>
#include <agx/BitArray.h>

#include <agx/Physics/RigidBodyEntity.h>

#include <agxCollide/Geometry.h>

#include <agx/SleepThreshold.h>
#include <agx/RigidBodyState.h>
#include <agx/SetVector.h>

#include <agx/macros.h>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4355) //Disable 'this' : used in base member initializer list warning
#endif

namespace agx
{
  class DynamicsSystem;
  class AutoSleep;
  class MergeSplitBodyStorage;
  class InternalData;

  AGX_DECLARE_POINTER_TYPES( RigidBody );
  AGX_DECLARE_VECTOR_TYPES( RigidBody );
  typedef agx::SetVector<ref_ptr<RigidBody> >    RigidBodyRefSetVector;

  /**
  The rigid body class, combining a geometric model and a frame of reference.
  */
  class CALLABLE AGXPHYSICS_EXPORT RigidBody : public agx::Referenced, public virtual agxStream::Serializable
  {
    public:
      /**
      The MotionControl enumeration indicates what makes a RigidBody move.

      There are three forms of motion allowed:
      -  STATIC  means no motion whatsoever;
      -  KINEMATICS means that motion is scripted;
      -  DYNAMICS means that motion results from forces;
      */
      enum MotionControl
      {
        STATIC     = 1, /**< This body will _never_ move. */
        KINEMATICS = 2, /**< This body's motion is scripted. (Position/velocity set by the user)*/
        DYNAMICS   = 3  /**< This body moves from the influence of forces. (Position/velocity updated by the system) */
      };

    public:
      /**
      Construct given name. Default: mass = 1, inertia diagonal = (1, 1, 1),
      geometries = 0, position = (0, 0, 0), linear velocity = (0, 0, 0),
      angular velocity = (0, 0, 0), enable = true, motion control = DYNAMICS.
      \param name - name of this rigid body (default: "")
      */
      RigidBody( const agx::Name& name = "" );

      /**
      Construct given geometry. Mass properties will be updated
      given \p geometry.
      \param geometry - first geometry to add to this rigid body
      */
      RigidBody( agxCollide::Geometry* geometry );

      /**
      Create a clone of the rigid body.
      */
      RigidBody *clone( bool shallowCopyRenderData=false ) const;

      /**
      Returns the model frame containing model the transformation and
      utilities to manipulate position, rotation etc.
      \return model frame of this rigid body
      */
      agx::Frame* getFrame();

      /**
      Returns the model frame containing model the transformation and
      utilities to manipulate position, rotation etc.
      \return model frame of this rigid body
      */
      const agx::Frame* getFrame() const;

      /**
      Set the transform of the body. The model frame will move to the specified transform,
      which is given in world coordinate frame.
      \param matrix - desired transform for the model frame in world coordinates.
      */
      void setTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return - the local transformation matrix of the body's model frame, relative to the parent.
      */
      const agx::AffineMatrix4x4& getLocalTransform() const;

      /**
      Assign the local transformation matrix for this body, ignoring any eventual parent transformation.
      \param matrix - transformation matrix relative to parent transform for the body's model frame
      */
      void setLocalTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return - the relative translate to the parent frame of the body's model frame.
      */
      agx::Vec3 getLocalPosition() const;

      /**
      Set the position of the body relative to its model frame's parent frame.
      \param p - local translate of body as a 3D vector
      */
      void setLocalPosition( const agx::Vec3& p );

      /**
      Set the position of the body relative to its model frame's parent frame.
      \param x - local x translate
      \param y - local y translate
      \param z - local z translate
      */
      void setLocalPosition( agx::Real x, agx::Real y, agx::Real z );

      /**
      Set the rotation of the body relative to world frame.
      \param q - rotation given as a quaternion
      */
      void setRotation( const agx::Quat& q );

      /**
      Set the rotation of the body relative to world frame.
      \param e - rotation given as Euler angles
      */
      void setRotation( const agx::EulerAngles& e);

      /**
      Set the rotation of the body relative to world frame.
      \param m - rotation given as an orthogonal transformation matrix
      */
      void setRotation( const agx::OrthoMatrix3x3& m );

      /**
      \return - the body's rotation relative to its model frame's parent frame.
      */
      agx::Quat getLocalRotation() const;

      /**
      Set the rotation of the body relative to its model frame's parent frame.
      \param q - rotation given as a quaternion
      */
      void setLocalRotation( const agx::Quat& q );

      /**
      Set the rotation of the body relative to its model frame's parent frame.
      \param e - rotation given as Euler angles
      */
      void setLocalRotation( const agx::EulerAngles& e );

      /**
      \return - the parent frame of the body's model frame, or 0 if this frame has no parent
      */
      agx::Frame* getParentFrame();

      /**
      \return - the parent frame of the body's model frame, or 0 if this frame has no parent
      */
      const agx::Frame* getParentFrame() const;

      /**
      Set the parent frame of this body's model frame.

      \note This method is not supposed to be used in general. It can result in a
      bad/erratic behaviour. Only for internal use.

      This means that getLocalTranslate, getLocalRotate, getLocalTransform will be given
      in the parents coordinate frame. I.e., this frame's transformation
      will be concatenated with the parents.
      \param frame - new parent frame, 0 to remove parent
      \return true if parent is changed (not the same as before) - otherwise false
      */
      bool setParentFrame( agx::Frame* frame );

      /**
      Set the position of the model frame in world coordinates.
      \param p - desired position in world coordinates.
      */
      void setPosition( const agx::Vec3& p );

      /**
      Set the position of the model frame in world coordinates.
      \param x - desired x-coordinate in world frame
      \param y - desired y-coordinate in world frame
      \param z - desired z-coordinate in world frame
      */
      void setPosition( agx::Real x, agx::Real y, agx::Real z );

      /**
      Return the center of mass (CM) frame of this rigid body.
      The CM frame is a child frame of the model frame. The
      center of mass offset is automatically calculated as
      geometries are added/removed, but this can be disabled
      in the mass properties of the rigid body.
      \return the center of mass frame of this rigid body
      */
      agx::Frame* getCmFrame();

      /**
      Return the center of mass (CM) frame of this rigid body.
      The CM frame is a child frame of the model frame. The
      center of mass offset is automatically calculated as
      geometries are added/removed, but this can be disabled
      in the mass properties of the rigid body.
      \return the center of mass frame of this rigid body
      */
      const agx::Frame* getCmFrame() const;

      /**
      Current model frame transform, given in world coordinate frame.
      \return the current model transformation of this rigid body, given in world coordinate frame
      */
      const agx::AffineMatrix4x4& getTransform() const;

      /**
      Current model frame position, given in world coordinate frame.
      \return the current model frame position, given in world coordinate frame
      */
      CALLABLE_UNIT("m/s")
      agx::Vec3 getPosition() const;

      /**
      Current model frame rotation, given in world coordinate frame.
      \return the current model frame rotation, given in world coordinate frame
      */
      agx::Quat getRotation() const;

      /**
      Set the linear velocity of the center of mass of this rigid body.
      This is the preferred way of moving KINEMATIC rigid bodies (a more
      convenient way is using moveTo(...)).
      The use for DYNAMIC is discouraged except for setting an initial velocity.
      If motion control is set to STATIC then the linear velocity will be
      set to zero - independent of the value of \p velocity.
      This method is seen as an instantaneous change in velocity (infinite acceleration),
      and will therefore not be represented by getAcceleration().
      \param velocity - velocity given in world coordinate frame
      */
      void setVelocity( const agx::Vec3& velocity );

      /**
      Set the linear velocity of the center of mass of this rigid body.
      This is the preferred way of moving KINEMATIC rigid bodies (a more
      convenient way is using moveTo(...)).
      The use for DYNAMIC is discouraged except for setting an initial velocity.
      If motion control is set to STATIC then the linear velocity will be
      set to zero - independent of the value of \p velocity.
      This method is seen as an instantaneous change in velocity (infinite acceleration),
      and will therefore not be represented by getAcceleration().
      \param vx,vy,vz - velocity given in world coordinate frame
      */
      void setVelocity( agx::Real vx, agx::Real vy, agx::Real vz );

      /**
      Set the angular velocity of the center of mass of this rigid body.
      This is the preferred way of moving KINEMATIC rigid bodies (a more
      convenient way is using moveTo(...)).
      The use for DYNAMIC is discouraged except for setting an initial angular velocity.
      If motion control is set to STATIC then the angular velocity will be
      set to zero - independent of the value of \p angularVelocity.
      This method is seen as an instantaneous change in angular velocity (infinite acceleration),
      and will therefore not be represented by getAngularAcceleration().
      \param angularVelocity - angular velocity given in world coordinate frame
      */
      void setAngularVelocity( const agx::Vec3& angularVelocity );

      /**
      Set the angular velocity of the center of mass of this rigid body.
      This is the preferred way of moving KINEMATIC rigid bodies (a more
      convenient way is using moveTo(...)).
      The use for DYNAMIC is discouraged except for setting an initial angular velocity.
      If motion control is set to STATIC then the angular velocity will be
      set to zero - independent of the value of \p angularVelocity.
      This method is seen as an instantaneous change in angular velocity (infinite acceleration),
      and will therefore not be represented by getAngularAcceleration().
      \param vx,vy,vz - angular velocity given in world coordinate frame
      */
      void setAngularVelocity( agx::Real vx, agx::Real vy, agx::Real vz );

      /**
      Velocity of center of mass frame origin, in world coordinate frame.
      \sa getModelVelocity
      \return linear velocity of the center of mass frame origin, in world coordinate frame
      */
      agx::Vec3 getVelocity() const;

      /**
      Angular velocity in world coordinate frame.
      \note Angular velocity is independent of the point it is measured
            in. I.e., methods like getModelAngularVelocity are redundant.
      \return angular velocity in world coordinate frame
      */
      agx::Vec3 getAngularVelocity() const;

      /**
      Calculates the linear velocity at the given point \p relPos. The
      point \p relPos must be given in model coordinate frame. The
      resulting linear velocity is given in world coordinate frame.
      \sa getModelAcceleration, getVelocity
      \param relPos - relative position given in model frame coordinates
      \return linear velocity in world frame of the point \p relPos
      */
      agx::Vec3 getModelVelocity( const agx::Vec3& relPos = agx::Vec3() ) const;

      /**
      Calculates the linear acceleration at the given point \p relPos. The
      point \p relPos must be given in model coordinate frame. The resulting
      linear acceleration is given in world coordinate frame.
      \sa getAcceleration, getModelVelocity
      \param relPos - relative position given in model frame coordinates
      \return linear acceleration in world frame of the point \p relPos
      */
      agx::Vec3 getModelAcceleration( const agx::Vec3& relPos = agx::Vec3() ) const;

      /**
      Change state enable of this rigid body.
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
      \return the motion control state of this rigid body
      */
      agx::RigidBody::MotionControl getMotionControl() const;

      /**
      Assign new motion control state to this rigid body. If
      new state is STATIC - linear- and angular velocity will
      be set to zero.
      Default: DYNAMICS
      \param control - new motion control
      */
      void setMotionControl( agx::RigidBody::MotionControl control );

      /**
      \return a pointer to the DynamicsSystem this rigid body belongs to
      */
      agx::DynamicsSystem* getSystem();

      /**
      \return a pointer to the DynamicsSystem this rigid body belongs to
      */
      const agx::DynamicsSystem* getSystem() const;

      /**
      Access the summed force applied to this rigid body last
      time the system was integrated.
      \sa addForce, addForceAtPosition
      \note Forces applied by constraints and/or contacts are
            NOT included in this value.
      */
      agx::Vec3 getLastForce() const;

      /**
      Access the summed torque applied to this rigid body last
      time the system was integrated.
      \sa addTorque
      \note Torques applied by constraints and/or contacts are
            NOT included in this value.
      */
      agx::Vec3 getLastTorque() const;

      /**
      Access property container of this rigid body. This method
      will create a new property container object if one hasn't
      been assigned before. Hence it is recommended to use
      hasPropertyContainer before calling this method.
      \note This method will create a new property container
            object if one hasn't been assigned earlier.
      \sa hasPropertyContainer
      \return a reference to the property container
      */
      agx::PropertyContainer* getPropertyContainer() const;

      /**
      \return true if it has an initialized property container
      */
      bool hasPropertyContainer() const;

      /**
      Assign new property container.
      \param container - property container
      */
      void setPropertyContainer( agx::PropertyContainer* container );

      /**
      \return the name of this rigid body
      */
      const agx::Name& getName() const;

      /**
      Assign new name to this rigid body.
      Default: ""
      */
      void setName( const agx::Name& name );

      /**
      Internal method. The id this rigid body has in a system. This
      value may change when any other body is removed from a system.
      \sa getUniqueID
      \return id to this rigid body in a system
      */
      agx::UInt32 getId() const;

      /**
      Returns custom auto-sleep parameters for this rigid body.
      Auto-sleep threshold is by default 0 (global/default settings).
      \sa setAutoSleepThreshold
      \return an AutoSleepThreshold if one has been assigned - otherwise 0
      */
      agx::AutoSleepThreshold* getAutoSleepThreshold();

      /**
      Returns custom auto-sleep parameters for this rigid body.
      Auto-sleep threshold is by default 0 (global/default settings).
      \sa setAutoSleepThreshold
      \return an AutoSleepThreshold if one has been assigned - otherwise 0
      */
      const agx::AutoSleepThreshold* getAutoSleepThreshold() const;

      /**
      Assign custom auto-sleep parameters to this rigid body.
      This object may be shared with other bodies.
      Default: 0 (global settings)
      \param threshold - the auto-sleep threshold
      */
      void setAutoSleepThreshold( agx::AutoSleepThreshold* threshold );

      /**
      Returns custom merge-split parameters for this rigid body.
      Merge-split threshold object is by default 0 (global/default settings).
      \return merge-split threshold used or 0 (global/default settings)
      */
      agx::MergeSplitThreshold* getMergeSplitThreshold();

      /**
      Returns custom merge-split parameters for this rigid body.
      Merge-split threshold object is by default 0 (global/default settings).
      \return merge-split threshold used or 0 (global/default settings)
      */
      const agx::MergeSplitThreshold* getMergeSplitThreshold() const;

      /**
      Assign custom merge-split parameters to this rigid body.
      This object may be shared with other bodies.
      Default: 0 (global settings)
      \param threshold - the merge-split threshold
      */
      void setMergeSplitThreshold( agx::MergeSplitThreshold* threshold );

      /**
      Convenience methods for center of mass frame. The bool flag
      of the mutator(set) methods indicates if the update should
      sync back to the model frame (i.e. the model is updated using
      the relative change in center of mass while the local center
      of mass values are unchanged), or if the update only affects
      the local center of mass frame (which is analogous to the
      normal frame interface).
      */

      /**
      \return the center of mass transform in world coordinate frame
      */
      const agx::AffineMatrix4x4& getCmTransform() const;

      /**
      Assign new center of mass transform, given in world coordinate
      frame. By default the model frame will be updated as well (\p synchronizeModel).
      \note The center of mass frame MUST, by definition, have the same
      orientation as the model frame. I.e., it is not defined to
      call this method with synchronizeModel = false and matrix having a different rotation than
      the model frame matrix.
      \param matrix - new matrix transform given in world coordinate frame
      \param synchronizeModel - default true, if false the model frame will not be updated with this new transform.
      */
      void setCmTransform( const agx::AffineMatrix4x4& matrix, bool synchronizeModel = true );

      /**
      \return the local offset of the center of mass position to the model origin (in model frame coordinates).
      */
      agx::Vec3 getCmLocalTranslate() const;

      /**
      Sets the local offset of the center of mass position to the model origin (in model frame coordinates).
      \param translate The new offset.
      */
      void setCmLocalTranslate(const agx::Vec3& translate);

      /**
      \return the center of mass position in world coordinate frame
      */
      agx::Vec3 getCmPosition() const;

      /**
      Assign new center of mass position, given in world coordinate
      frame. By default the model frame will be updated as well (\p synchronizeModel).
      \param p - new center of mass position given in world coordinate frame
      \param synchronizeModel - default true, if false the model frame will not be updated with this new position
      */
      void setCmPosition( const agx::Vec3& p, bool synchronizeModel = true );

      /**
      \return the center of mass rotation in world coordinate frame
      */
      agx::Quat getCmRotation() const;

      /**
      Assign new center of mass rotation, given in world coordinate
      frame.
      \note The center of mass frame MUST, by definition, have the same
            orientation as the model frame. I.e., it is not defined to
            call this method with synchronizeModel = false.
      \param q - new center of mass rotation (quaternion)
      */
      void setCmRotation( const agx::Quat& q);

      /**
      Assign new center of mass rotation, given in world coordinate
      frame.
      \note The center of mass frame MUST, by definition, have the same
            orientation as the model frame. I.e., it is not defined to
            call this method with synchronizeModel = false.
      \param e - new center of mass rotation (Euler angles)
      */
      void setCmRotation( const agx::EulerAngles& e);

      /**
      Assign new center of mass rotation, given in world coordinate
      frame.
      \note The center of mass frame MUST, by definition, have the same
            orientation as the model frame. I.e., it is not defined to
            call this method with synchronizeModel = false.
      \param m - new center of mass rotation (Orthonormal matrix)
      */
      void setCmRotation( const agx::OrthoMatrix3x3& m);

      /**
      Convenience method for setting the inertia tensor from extern
      calculations given in local center of mass coordinate frame.
      \param inertiaTensor - the externally computed inertia tensor
                             given in local center of mass frame
      \param cmLocalTransform - the frame in which the inertia tensor
                                is given (optional, default: Identity)
      \param autogenerateCmOffset - if true, the center of mass offset will
                                    be recalculated when the geometry configuration
                                    of the body changes (optional, default: false)
      \param autogenerateInertia - if set to true, the inertia tensor will be
                                   recalculated when the geometry configuration
                                   of the body changes (optional, default: false)
      */
      void setInertiaTensor( const agx::SPDMatrix3x3& inertiaTensor, const agx::AffineMatrix4x4& cmLocalTransform = agx::AffineMatrix4x4(), bool autogenerateCmOffset = false, bool autogenerateInertia = false );

      /**
      Convenience method for setting the inertia tensor from extern calculations,
      in diagonal form, given in local center of mass coordinate frame.
      \param inertiaDiagonal - the externally computed inertia tensor in diagonal
                               form given in local center of mass coordinate frame
                               defined by \p cmLocalTransform
      \param cmLocalTransform - the frame in which the inertia tensor
                                is given (optional, default: Identity)
      \param autogenerateCmOffset - if true, the center of mass offset will
                                    be recalculated when the geometry configuration
                                    of the body changes (optional, default: false)
      \param autogenerateInertia - if set to true, the inertia tensor will be
                                   recalculated when the geometry configuration
                                   of the body changes (optional, default: false)
      */
      void setInertiaTensor( const agx::Vec3& inertiaDiagonal, const agx::AffineMatrix4x4& cmLocalTransform = agx::AffineMatrix4x4(), bool autogenerateCmOffset = false, bool autogenerateInertia = false );

      /**
      Add a named attachment frame to this rigid body.
      \return false if the name is non-unique, true if successfully added
      */
      bool addAttachment( agx::Frame* frame, const agx::String& name );

      /**
      Remove an attachment from the rigid body.
      \return false if the named attachment could not be found, true if successfully removed
      */
      bool removeAttachment( const agx::String& name );

      /**
      Access a named attachment frame.
      \param name - name of the attachment
      \return the attachment frame with \p name if it exist - otherwise null
      */
      Frame* getAttachment( const agx::String& name );

      /**
      \return mass related properties of this rigid body
      */
      const agx::MassProperties* getMassProperties() const;

      /**
      \return mass related properties of this rigid body
      */
      agx::MassProperties* getMassProperties();

      /**
      Explicitly set the force, given in world coordinate frame, that will be
      affecting this body in the next solve.
      \note Force added before calling this method will be overwritten. Force added
           after calling this method will be added to \p force.
      \param force - the force
      */
      void setForce( const agx::Vec3& force );

      /**
      Explicitly set the force, given in world coordinate frame, that will be
      affecting this body in the next solve.
      \note Force added before calling this method will be overwritten. Force added
           after calling this method will be added to \p force.
      \param fx,fy,fz - the force
      */
      void setForce( agx::Real fx, agx::Real fy, agx::Real fz );

      /**
      Explicitly set the torque, given in world coordinate frame, that will be affecting
      this body in the next solve.
      \note Torque added before calling this method will be overwritten. Torque added
            after calling this method will be added to \p torque.
      \param torque - the torque
      */
      void setTorque( const agx::Vec3& torque );

      /**
      Explicitly set the torque, given in world coordinate frame, that will be affecting
      this body in the next solve.
      \note Torque added before calling this method will be overwritten. Torque added
            after calling this method will be added to \p torque.
      \param tx,ty,tz - the torque
      */
      void setTorque( agx::Real tx, agx::Real ty, agx::Real tz );

      /**
      Add the specified \p force, given in world coordinate frame, that will be
      affecting this body in the next solve. The force will be applied to the Center of Mass.
      \param force - the force, given in world coordinate frame
      */
      void addForce( const agx::Vec3& force );

      /**
      Add the specified force,  \p fx \p fy \p fz, given in world coordinate frame, that will be
      affecting this body in the next solve. The force will be applied to the Center of Mass.
      \param fx, fy, fz - the force, given in world coordinate frame
      */
      void addForce( agx::Real fx, agx::Real fy, agx::Real fz );

      /**
      Add a force, given in the World coordinate frame, applied at a point specified in world coordinate frame. If the position
      is different from the center of mass, a torque will be calculated and added as well.
      \param force - the force, given in world coordinate frame, to be added to the already previously added forces
      \param position - the position given in world coordinate frame where the force should be applied
      */
      void addForceAtPosition( const agx::Vec3& force, const agx::Vec3& position );

      /**
      Add a force, given in the world coordinate frame, applied at a point specified in the world coordinate frame. If the position
      is different from the center of mass, a torque will be calculated and added as well.
      \param fx,fy,fz - the force, given in world coordinate frame, to be added to the already previously added forces
      \param px,py,pz - the position given in world coordinate frame where the force should be applied
      */
      void addForceAtPosition( agx::Real fx, agx::Real fy, agx::Real fz, agx::Real px, agx::Real py, agx::Real pz );

      /**
      Add a force, given in the world coordinate frame, applied at a point specified in the local Model coordinate frame.
      If the position is different from the center of mass, a torque will be calculated and added as well.
      \param force - the force, given in world coordinate frame, to be added to the already previously added forces
      \param position - the position given in local coordinate frame where the force should be applied
      */
      void addForceAtLocalPosition( const agx::Vec3& force, const agx::Vec3& position );

      /**
      Add a force given in world coordinate frame, applied at a point specified in local coordinate model frame.
      If the position is different from the center of mass, a torque will be calculated and added as well.
      \param fx,fy,fz - the force, given in world coordinate frame, to be added to the already previously added forces
      \param px,py,pz - the position given in local coordinate frame where the force should be applied
      */
      void addForceAtLocalPosition( agx::Real fx, agx::Real fy, agx::Real fz, agx::Real px, agx::Real py, agx::Real pz );

      /**
      Add a force given in world coordinate frame, applied at a point specified in the local Center of Mass coordinate frame.
      If the position is different from (0,0,0) (CoM), this method will also introduce a torque.
      \param force - the force, given in world coordinate frame, to be added to the already previously added forces
      \param position - the position in local center of mass coordinate frame where the force should be applied
      */
      void addForceAtLocalCmPosition( const agx::Vec3& force, const agx::Vec3& position );

      /**
      Add a force given in world coordinate frame, applied at a point specified in the local Center of Mass coordinate frame.
      If the position is different from (0,0,0) (CoM), this method will also introduce a torque.
      \param fx,fy,fz - the force, given in world coordinate frame, to be added to the already previously added forces
      \param px,py,pz - the position in local center of mass coordinate frame where the force should be applied
      */
      void addForceAtLocalCmPosition( agx::Real fx, agx::Real fy, agx::Real fz, agx::Real px, agx::Real py, agx::Real pz );

      /**
      Adds the specified \p torque, given in world coordinate frame, that will be
      affecting this body in the next solve.
      \param torque - the torque, given in world coordinate frame
      */
      void addTorque( const agx::Vec3& torque );

      /**
      Adds the specified \p torque, given in world coordinate frame, that will be
      affecting this body in the next solve.
      \param tx,ty,tz - the torque, given in world coordinate frame
      */
      void addTorque( agx::Real tx, agx::Real ty, agx::Real tz );

      /**
      Adds the specified torque, given in Center of Mass coordinate frame that will be
      affecting this body in the next solve.
      \param tx,ty,tz - the torque, given in local coordinate frame
      */
      void addLocalTorque( agx::Real tx, agx::Real ty, agx::Real tz );

      /**
      Adds the specified \p torque, given in Center of Mass coordinate frame that will be
      affecting this body in the next solve.
      \param torque - the torque
      */
      void addLocalTorque( const agx::Vec3& torque );

      /**
      Connect a geometry to this rigid body. The frame of the geometry will be attached as a child
      to the model frame of the rigid body. The inertia, mass, and center of mass offset will be automatically
      recalculated using the material and volume of the geometry. However, any parameter explicitly set in the
      MassProperties (with auto generation flag disabled) will _not_ be recalculated/updated.

      \note If the Geometry explicitly has to be added to the Simulation, adding it to a RigidBody (part of a simulation)
      will NOT add the geometry to the same simulation.

      \note Regarding \p incrementalMassCalculation: Incremental mass properties calculation is computationally
            less expensive but may cause numerical drift if this method is used many times. If false, the
            current set of geometries will be used to calculate the mass properties, which in some extreme
            cases can be more computationally expensive (or cause general overhead).
      \param geometry - the geometry to add
      \param localTransform - the relative transform to where the geometry should be placed relative
                              model coordinate frame of this rigid body
      \param incrementalMassCalculation - false for full recalculation of mass properties, true for incremental calculation (see note)
      \return true if successful - false if not added (0, geometry is part of another rigid body or already added)
      */
      bool add( agxCollide::Geometry* geometry , const agx::AffineMatrix4x4& localTransform, bool incrementalMassCalculation = false );

      /**
      Add a geometry to this rigid body. The frame of the geometry will be attached as a child
      to the model frame of this rigid body. The inertia, mass, and center of mass offset will be automatically
      recalculated using the material and volume of the geometry. However, any parameter explicitly set in the
      MassProperties (with auto generation flag disabled) will _not_ be recalculated/updated.
      The local transformation in the geometry (geometry->getLocalTransform()) will be used to transform the geometry
      relative to the RigidBody.

      \note If the Geometry explicitly has to be added to the Simulation, adding it to a RigidBody (part of a simulation)
      will NOT add the geometry to the same simulation.

      \note Regarding \p incrementalMassCalculation: Incremental mass properties calculation is computationally
            less expensive but may cause numerical drift if this method is used many times. If false, the
            current set of geometries will be used to calculate the mass properties, which in some extreme
            cases can be more computationally expensive (or cause general overhead).
      \note Currently, automatic mass property calculation (incremental or not) assumes all geometries and shapes to
            be disjoint (non-overlapping). If they overlap, the effect of this assumption is that the overlapping
            volume will be counted twice into mass and inertia computation. It is recommended to set the mass properties manually in
            this case.
      \param geometry - the geometry to add
      \param incrementalMassCalculation - false for full recalculation of mass properties, true for incremental calculation (see note)
      \return false if the given geometry already has a body associated to it and override is set to false.
      */
      bool add( agxCollide::Geometry* geometry, bool incrementalMassCalculation = false );

      /**
      Remove geometry if \p geometry is part of this rigid body.
      \note Regarding \p incrementalMassCalculation: Incremental mass properties calculation is computationally
            less expensive but may cause numerical drift if this method is used many times. If false, the
            current set of geometries will be used to calculate the mass properties, which in some extreme
            cases can be more computationally expensive (or cause general overhead).
      \param geometry - geometry to remove
      \param incrementalMassCalculation - false for full recalculation of mass properties, true for incremental calculation (see note)
      \return true if geometry removed - otherwise false (e.g., geometry not part of this rigid body)
      */
      bool remove( agxCollide::Geometry* geometry, bool incrementalMassCalculation = false );

      /**
      Find (linear search) the first Geometry in the RigidBody with a matching name
      \param name - name of geometry to search for
      \return a pointer to the found Geometry, null if not found.
      */
      const agxCollide::Geometry* getGeometry(const agx::Name& name) const;

      /**
      Find (linear search) the first Geometry in the RigidBody with a matching name
      \param name - name of geometry to search for

      \return a pointer to the found Geometry, null if not found.
      */
      agxCollide::Geometry* getGeometry(const agx::Name& name);

      /**
      \return container with all geometries that is part of this rigid body
      */
      const agxCollide::GeometryRefVector& getGeometries() const;

      /**
      Merge-split related.
      \return the geometries this body originally had (merge-split may add or remove geometries)
      */
      const agxCollide::GeometryRefVector& getOriginalGeometries() const;

      /**
      Utility method to calculate and assign linear- and angular velocity given a target transform and
      the time it should take to get there. This is basically only valid if motion control is set
      to KINEMATICS.
      \note This rigid body will coincide with the target transform given global model frame transform.
      \note This rigid body will not stop at target transform after time t.
      \param target - target transform given in world coordinate frame
      \param t - time this rigid body has to reach the target
      */
      void moveTo( const agx::AffineMatrix4x4& target, agx::Real t );

      /**
      Utility method to calculate and assign linear- and angular velocity given a target transform and
      the time it should take to get there. This is basically only valid if motion control is set
      to KINEMATICS.
      \note This rigid body will coincide with the target transform given global model frame transform.
      \note This rigid body will not stop at target transform after time t.
      \param targetPosition - target position given in world coordinate frame
      \param targetRotation - target rotation given in world coordinate frame
      \param t - time this rigid body has to reach the target
      */
      void moveTo( const agx::Vec3& targetPosition, const agx::Quat& targetRotation, agx::Real t );

      /**
      Method to explicitly update mass properties. E.g., when the density is changed on
      a material which this rigid body depends on. This method uses the current mass
      property update mask. I.e., if this rigid body has an explicitly assigned mass,
      mass will not be updated during this call.
      */
      void updateMassProperties();

      /**
      Method to explicitly update mass properties. E.g., when the density is changed on
      a material which this rigid body depends on. This method takes the mass property
      update mask as argument. I.e., if this rigid body has an explicitly assigned mass,
      and \p mask contains agx::MassProperties::MASS, a new mass will be calculated and
      assigned.
      \note The mask passed to this method will not be stored. I.e., the mass property
            update state is the same before and after calling this method.
      \param mask - bit mask of agx::MassProperties::AutoGenerateFlags which mass properties that should be updated
      */
      void updateMassProperties( agx::UInt32 mask );

      /**
      Calculate the mass of this rigid body using the volume and density of added geometries.
      \note This calculated mass isn't necessary the same as getMassProperties()->getMass()
            since the mass in mass properties could be assigned (i.e., explicit).
      \return the total mass of this rigid body, calculated given volume and density of added geometries
      */
      agx::Real calculateMass() const;

      /**
      Calculate the center of mass offset of this rigid body using the volume and density of
      the added geometries.
      \param mass - if 0, the mass will be recalculated given the added geometries density and volume
      \return the center of mass offset given in local model coordinate frame
      */
      agx::Vec3 calculateCenterOfMassOffset( agx::Real mass = agx::Real() ) const;

      /**
      Specify a vector (in body coordinates) along which no linear velocity damping should occur.
      Default: (0, 0, 0)
      \param dir - direction, given in local body coordinate frame, in which the linear damping should be zero
      */
      void setLinearVelocityZeroDamping( const agx::Vec3f& dir );

      /**
      Specify a vector (in body coordinates) along which no angular velocity damping should occur.
      Default: (0, 0, 0)
      \param dir - direction, given in local body coordinate frame, in which the angular damping should be zero
      */
      void setAngularVelocityZeroDamping( const agx::Vec3f& dir );

      /**
      \return the zero damping vector (in local body coordinate frame) which points out the direction
              in which no linear velocity damping should be applied
      */
      agx::Vec3f getLinearVelocityZeroDamping() const;

      /**
      \return the zero damping vector (in local body coordinate frame) which points out the direction
              in which no angular velocity damping should be applied
      */
      agx::Vec3f getAngularVelocityZeroDamping() const;

      /**
      Set linear velocity damping for the body in all directions x, y and z.
      The unit of this damping is mass over time.
      Default: 0 (no damping)
      \param damping - the damping
      */
      void setLinearVelocityDamping( float damping );

      /**
      Give linear velocity damping in each direction, given in local body coordinate frame.
      The unit of this damping is mass over time.
      Default: (0, 0, 0) (no damping).
      \param damping - the damping
      */
      void setLinearVelocityDamping( const agx::Vec3f& damping );

      /**
      Set angular velocity damping for the body in all directions x, y and z.
      The unit of this damping is mass over time.
      Default: 0 (no damping)
      \param damping - the damping
      */
      void setAngularVelocityDamping( float damping );

      /**
      Give angular velocity damping in each direction, given in local body coordinate frame.
      The unit of this damping is mass over time.
      Default: (0, 0, 0) (no damping).
      \param damping - the damping
      */
      void setAngularVelocityDamping( const agx::Vec3f& damping );

      /**
      Set both linear- and angular velocity damping in all directions.
      \sa setLinearVelocityDamping, setAngularVelocityDamping
      \param damping - the damping
      */
      void setVelocityDamping( float damping );

      /**
      \return the linear velocity damping, given in local body coordinate frame
      */
      agx::Vec3f getLinearVelocityDamping() const;

      /**
      \return the linear velocity damping, given in local body coordinate frame
      */
      agx::Vec3f getAngularVelocityDamping() const;

      /**
      \return the linear momentum for this body
      */
      agx::Vec3 getLinearMomentum() const;

      /**
      \return the angular momentum for this body
      */
      agx::Vec3 getAngularMomentum() const;

      /**
      Set this, six degrees of freedom, rigid body to be handled as a three
      degrees of freedom particle (rotational degrees of freedom ignored).
      Default: false
      \param handleAsParticle - if true this rigid body will be handled as a particle
      */
      void setHandleAsParticle( bool handleAsParticle );

      /**
      \return true if this rigid body is handled like a particle (i.e., rotational degrees of freedom ignored)
      */
      bool getHandleAsParticle() const;

      /**
      Will always be 0 for KINEMATIC and STATIC bodies.
      \return the linear acceleration as the difference in linear velocity between this and the previous time step
      */
      agx::Vec3 getAcceleration() const;

      /**
      Will always be 0 for KINEMATIC and STATIC bodies.
      \return the angular acceleration as the difference in linear velocity between this and the previous time step
      */
      agx::Vec3 getAngularAcceleration() const;

      /**
      \note The number of constraints value is updated when the interaction
            graph is created during the solve process. This means that there
            are a one time step delay on this value.
      \return true if the number of constraints (contacts excluded) was larger than zero, the last time step
      */
      bool hasConstraints() const;

      /**
      \note The number of constraints value is updated when the interaction
            graph is created during the solve process. This means that there
            are a one time step delay on this value.
      \return the number of constraints (contacts excluded) connected to this rigid body, the last time step
      */
      agx::UInt16 getNumConstraints() const;

      /**
      \sa setCustomData
      \return user data, if assigned
      */
      agx::Referenced* getCustomData() const;

      /**
      It's possible to derive from agx::Referenced and implement a
      reference counted object and associate it to this rigid body
      via this method. If this rigid body is the only one holding
      a reference to the data, the data will be deleted when this
      rigid body is deleted.
      \sa getCustomData
      \param data - custom data
      */
      void setCustomData( agx::Referenced* data );

      /**
      Internal method. Data access.
      \return the rigid body entity
      */
      agx::Physics::RigidBodyPtr getEntity() const;

      /**
      Internal method. Data access.
      \param entity - new rigid body entity
      */
      void setEntity( agx::Physics::RigidBodyPtr entity );

      /**
      Internal method. Transfer data from one storage to another.
      \param storage - new storage
      */
      void transfer( agxData::EntityStorage* storage );

      /**
      \return the current external force applied to the body using the method(s) add/setForce
      */
      agx::Vec3 getForce() const;

      /**
      \return the current external torque applied to the body using the method(s) add/setTorque
      */
      agx::Vec3 getTorque() const;

      /**
      Internal method.
      \deprecated Instead we recommend using AMOR/MergeSplitHandler
      \return a merge-split body storage object, used when merge-split is enabled
      */
      agx::MergeSplitBodyStorage* getBodyStorage();

      /**
      Internal method.
      \deprecated Instead we recommend using AMOR/MergeSplitHandler
      \return a merge-split body storage object, used when merge-split is enabled
      */
      const agx::MergeSplitBodyStorage* getBodyStorage() const;

      /**
      \return true if body has a valid body storage, false if it is cleared (null)
      */
      bool hasBodyStorage() const;

      /**
      \return true if body was created by an agxModel::PHysicalDimension.
      */
      bool isPowerlineBody() const;

      /**
      Internal method.
      */
      void setIsPowerlineBody(bool isHydraulicBody);

      /**
      If a body (child) is merged into a (parent) body the child will
      not get its transform (getTransform, getPosition, getRotation)
      updated by the ordinary integration method (as it is disabled).
      Therefore if the world transformation of the child is needed,
      it can be accessed through this method. If the body is not
      merged, it will return getTransform().
      \return a calculated model world transform
      */
      agx::AffineMatrix4x4 getMergedWorldTransform() const;

      /**
      \return true if this body is merged (as parent or as child) to another body
      */
      bool isMerged() const;

#ifndef SWIG
      /// \return true if the specified body \p state should not be affected by a gravity field.
      inline static bool shouldIgnoreGravity(agx::RigidBodyState state)
      {
        return !state.enabled() ||
          state.motionControl() != agx::RigidBody::DYNAMICS ||
          state.powerline();
      }
#endif

      /**
      Data holder for merge-split properties.
      \deprecated Instead we recommend using AMOR/MergeSplitHandler
      */
      class AGXPHYSICS_EXPORT MergeSplitProperties
      {
        public:
          /**
          Default constructor. Merge enable: false, enable split: false,
          merge only with static: false.
          */
          MergeSplitProperties();

          /**
          Enable merge. Default: false
          \param flag - true to enable merge, false to disable
          */
          void setEnableMerge( bool flag );

          /**
          \return true if merge is enabled - otherwise false
          */
          bool getEnableMerge() const;

          /**
          Enable split. Default: true
          \param flag - true to enable split, false to disable
          */
          void setEnableSplit( bool flag );

          /**
          \return true if split is enabled - otherwise false
          */
          bool getEnableSplit() const;

          /**
          Enable merge only with static. Default: false
          \param flag - true to enable merge only with static, false to disable
          */
          void setMergeOnlyWithStatic( bool flag );

          /**
          \return true if merge only with static is enabled - otherwise false
          */
          bool getMergeOnlyWithStatic() const;

        private:
          enum FLAGS
          {
            ENABLE_MERGE,
            ENABLE_SPLIT,
            ENABLE_MERGE_WITH_STATIC,
            NUM_ELEMENTS
          };

          typedef agx::BitArray<NUM_ELEMENTS> FlagVector;

        private:
          FlagVector m_flags;
      };

      /**
      \return data for merge-split properties
      */
      agx::RigidBody::MergeSplitProperties& getMergeSplitProperties();
      const agx::RigidBody::MergeSplitProperties& getMergeSplitProperties() const;

      /**
      Data holder for auto-sleep properties.
      */
      class AGXPHYSICS_EXPORT AutoSleepProperties
      {
        public:
          /**
          Specifies the state of auto sleep for this body
          */
          enum State {
            AWAKE  = 0,          /**< Body is active */
            TIRED  = 1,          /**< Body is not moving and is about to be put to sleep by the system*/
            ASLEEP = 2           /**< Body has been put to sleep */
          };

          /**
          Put the body into sleep or wake it up.
          A body in sleep does not occupy time in the solver.
          It can be awaken by contact/separation or a velocity/acceleration > given threshold.

          If a body is put to sleep, it will get its velocities set to zero.
          It will be removed from the DynamicsSystem's simulated bodies

          \param flag - If true, the body will be put to sleep.
          */
          void setSleeping( bool flag );

          /**
          \return the current dynamic state of the RigidBody
          */
          State getState() const;

          /**
          \return true if body is sleeping
          */
          bool isSleeping(  ) const;

          /**
          Enable/Disable auto sleep functionality for this body.
          Only a body with auto sleep enabled can be put to sleep and awaken by the system.
          \param flag - if true, auto sleep is enabled for this body
          */
          void setEnable( bool flag );

          /**
          \return true if this body is enabled for auto sleep
          */
          bool getEnable( ) const;

          /**
          \return the time this body has been TIRED
          */
          float getRestTime() const;

        protected:
          // Never create this outside of a RigidBody.
          AutoSleepProperties( RigidBody *body = nullptr );

        private:
          /**
          Private method for putting the body into sleep or wake it up.
          A body in sleep does not occupy time in the solver.
          It can be awaken by contact/separation or a velocity/acceleration > given threshold.

          If a body is put to sleep, it will get its velocities set to zero.
          It will be removed from the DynamicsSystem's simulated bodies

          \param flag - If true, the body will be put to sleep.
          \param state - The new state which the body will get.
          */
          void setSleeping( bool flag, State state );

          /**
          Set the current dynamic state of the RigidBody
          \return true it state successfully was set, false if AutoSleep is disabled and we tried to sleep a body
          */
          void setState( State state );

          friend class RigidBody;
          friend class AutoSleep;
          friend class ConstraintImplementation;

          /// Set the time this body has been TIRED
          void setRestTime( float t );

        private:
          float m_restTime;
          bool m_enabled;
          agx::UInt8 m_state;
          RigidBody *m_body;
      };

      /**
      \return data for auto-sleep properties
      */
      agx::RigidBody::AutoSleepProperties& getAutoSleepProperties();
      const agx::RigidBody::AutoSleepProperties& getAutoSleepProperties() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::RigidBody );

      bool hasValidEntityIndex() const;

    protected:
      friend class DynamicsSystem;
      friend class GeometryListener;
      friend class ConstraintImplementation;
      friend class MergeSplit;
      friend class MergeSplitBodyStorage;
      friend class AutoSleepProperties;

      virtual ~RigidBody();

      // Never use these
      RigidBody( const agx::RigidBody& );
      RigidBody( const agx::Name& name, agx::Physics::RigidBodyPtr entity );
      RigidBody& operator = ( const agx::RigidBody& ) { return *this; }

      /**
      Assign dynamics system unique id to this rigid body.
      \param id - index to this rigid body in dynamics system
      */
      void setId( agx::UInt32 id );

      /**
      Assign dynamics system where this body belongs to.
      \param system - dynamics system this body belongs to
      */
      void setSystem( agx::DynamicsSystem* system );

      /**
      Give this body a storage for merge-split.
      \param bodyStorage - merge-split body storage for this rigid body
      */
      void setBodyStorage( agx::MergeSplitBodyStorage* bodyStorage );

      /**
      Clear the BodyStorage for this RigidBody. hasBodyStorage() will return false after this.
      */
      void clearBodyStorage();

      /**
      Will calculate new mass properties for this rigid body as the previously calculated one +/- the specified \p geometry
      \param mask - Specifies what will be updated
      \param sign - 1 means adding, -1 means removing the geometry.
      \param geometry - The geometry for which the mass properties will be added/removed
      */
      void incrementalMassPropertyUpdate( agx::UInt32 mask, int sign, const agxCollide::Geometry* geometry );

    private:
      void init( const agx::Name& name );

      void incrementNumConstraints();
      void decrementNumConstraints();
      void resetNumConstraints();

      void removeAllGeometries();
      void appendGeometry( agxCollide::Geometry* geometry );
      void removeGeometry( agx::Physics::GeometryPtr before, agx::Physics::GeometryPtr toBeRemoved );

      DOXYGEN_START_INTERNAL_BLOCK()
      friend class InternalData;
      /**
      \return internal data
      */
      agx::Referenced* getInternalData() const;

      /**
      Assign internal data.
      */
      void setInternalData( agx::Referenced* data );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      // Low level data.
      Physics::RigidBodyRef m_entity;

      mutable PropertyContainerRef m_propertyContainer;

      DynamicsSystem* m_mainSystem;

      AutoSleepThresholdRef m_autoSleepThreshold;
      MergeSplitThresholdRef m_mergeSplitThreshold;

      FrameRef m_frame;
      FrameRef m_cmFrame;

      MassProperties m_massProperties;

      agxCollide::GeometryRefVector m_geometries;

      typedef HashTable<agx::String, FrameRef > FrameTable;
      FrameTable m_namedAttachments;

      agx::ref_ptr<agx::Referenced> m_customData;
      agx::ref_ptr<agx::Referenced> m_internalData;

      MergeSplitBodyStorage* m_bodyStorage;
      MergeSplitProperties m_msProperties;
      AutoSleepProperties m_autoSleepProperties;
  };

  AGX_FORCE_INLINE bool RigidBody::getEnable() const
  {
    return m_entity.state().enabled();
  }

  AGX_FORCE_INLINE bool RigidBody::isEnabled() const
  {
    return m_entity.state().enabled();
  }

  AGX_FORCE_INLINE RigidBody::MotionControl RigidBody::getMotionControl() const
  {
    return (MotionControl)m_entity.state().motionControl();
  }

  AGX_FORCE_INLINE bool RigidBody::hasPropertyContainer() const
  {
    return ( m_propertyContainer.isValid() );
  }

  AGX_FORCE_INLINE agx::PropertyContainer* RigidBody::getPropertyContainer() const
  {
    if ( m_propertyContainer == nullptr )
      m_propertyContainer = new agx::PropertyContainer;
    return m_propertyContainer;
  }

  inline Physics::RigidBodyPtr RigidBody::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE UInt32 RigidBody::getId() const
  {
    return m_entity.id();
  }

  AGX_FORCE_INLINE AutoSleepThreshold* RigidBody::getAutoSleepThreshold()
  {
    return m_autoSleepThreshold;
  }

  AGX_FORCE_INLINE const AutoSleepThreshold* RigidBody::getAutoSleepThreshold() const
  {
    return m_autoSleepThreshold;
  }

  AGX_FORCE_INLINE MergeSplitThreshold* RigidBody::getMergeSplitThreshold()
  {
    return m_mergeSplitThreshold;
  }

  AGX_FORCE_INLINE const MergeSplitThreshold* RigidBody::getMergeSplitThreshold() const
  {
    return m_mergeSplitThreshold;
  }

  AGX_FORCE_INLINE Frame* RigidBody::getFrame()
  {
    return m_frame.get();
  }

  AGX_FORCE_INLINE const Frame* RigidBody::getFrame() const
  {
    return m_frame.get();
  }

  AGX_FORCE_INLINE Frame* RigidBody::getCmFrame()
  {
    return m_cmFrame.get();
  }

  AGX_FORCE_INLINE const Frame* RigidBody::getCmFrame() const
  {
    return m_cmFrame.get();
  }

  AGX_FORCE_INLINE const AffineMatrix4x4& RigidBody::getCmTransform() const
  {
    return m_entity.cmTransform();
  }

  AGX_FORCE_INLINE Vec3 RigidBody::getVelocity() const
  {
    return m_entity.velocity();
  }

  AGX_FORCE_INLINE Vec3 RigidBody::getAngularVelocity() const
  {
    return m_entity.angularVelocity();
  }

  AGX_FORCE_INLINE Vec3 RigidBody::getCmLocalTranslate() const
  {
    return getCmFrame()->getLocalTranslate();
  }

  AGX_FORCE_INLINE Vec3 RigidBody::getCmPosition() const
  {
    return getCmTransform().getTranslate();
  }

  AGX_FORCE_INLINE Quat RigidBody::getCmRotation() const
  {
    return getCmTransform().getRotate();
  }

  AGX_FORCE_INLINE MassProperties* RigidBody::getMassProperties()
  {
    return &m_massProperties;
  }

  AGX_FORCE_INLINE const MassProperties* RigidBody::getMassProperties() const
  {
    return &m_massProperties;
  }

  AGX_FORCE_INLINE bool RigidBody::getHandleAsParticle() const
  {
    return m_entity.state().handleAsParticle();
  }

  AGX_FORCE_INLINE bool RigidBody::MergeSplitProperties::getEnableMerge() const
  {
    return m_flags[ENABLE_MERGE];
  }

  AGX_FORCE_INLINE bool RigidBody::MergeSplitProperties::getEnableSplit() const
  {
    return m_flags[ENABLE_SPLIT];
  }

  AGX_FORCE_INLINE bool RigidBody::MergeSplitProperties::getMergeOnlyWithStatic() const
  {
    return m_flags[ENABLE_MERGE_WITH_STATIC];
  }

  AGX_FORCE_INLINE RigidBody::MergeSplitProperties& RigidBody::getMergeSplitProperties()
  {
    return m_msProperties;
  }

  AGX_FORCE_INLINE const RigidBody::MergeSplitProperties& RigidBody::getMergeSplitProperties() const
  {
    return m_msProperties;
  }

  AGX_FORCE_INLINE RigidBody::AutoSleepProperties::State RigidBody::AutoSleepProperties::getState() const
  {
    return (State)m_state;
  }

  AGX_FORCE_INLINE bool RigidBody::AutoSleepProperties::getEnable() const
  {
    return m_enabled;
  }

  AGX_FORCE_INLINE bool RigidBody::AutoSleepProperties::isSleeping() const
  {
    return getState() == ASLEEP;
  }

  AGX_FORCE_INLINE float RigidBody::AutoSleepProperties::getRestTime() const
  {
    return m_restTime;
  }

  AGX_FORCE_INLINE void RigidBody::AutoSleepProperties::setRestTime( float t )
  {
    m_restTime = t;
  }

  AGX_FORCE_INLINE RigidBody::AutoSleepProperties& RigidBody::getAutoSleepProperties()
  {
    return m_autoSleepProperties;
  }
  AGX_FORCE_INLINE const RigidBody::AutoSleepProperties& RigidBody::getAutoSleepProperties() const
  {
    return m_autoSleepProperties;
  }

  AGX_FORCE_INLINE Vec3 RigidBody::getAcceleration() const
  {
    return m_entity.linearAcceleration();
  }

  AGX_FORCE_INLINE Vec3 RigidBody::getAngularAcceleration() const
  {
    return m_entity.angularAcceleration();
  }

  AGX_FORCE_INLINE bool RigidBody::hasConstraints() const
  {
    return m_entity.numConstraints() > 0;
  }

  AGX_FORCE_INLINE UInt16 RigidBody::getNumConstraints() const
  {
    return m_entity.numConstraints();
  }

  AGX_FORCE_INLINE Referenced* RigidBody::getCustomData() const
  {
    return m_customData.get();
  }

  AGX_FORCE_INLINE const agx::AffineMatrix4x4& RigidBody::getTransform() const
  {
    return m_entity.modelTransform();
  }

  AGX_FORCE_INLINE agx::Vec3 RigidBody::getPosition() const
  {
    return m_entity.modelTransform().getTranslate();
  }

  AGX_FORCE_INLINE agx::Quat RigidBody::getRotation() const
  {
    return m_entity.modelTransform().getRotate();
  }

  inline Referenced* RigidBody::getInternalData() const
  {
    return m_internalData;
  }

  inline void RigidBody::setVelocity( agx::Real vx, agx::Real vy, agx::Real vz )
  {
    setVelocity(agx::Vec3(vx,vy,vz));
  }

  inline void RigidBody::setAngularVelocity( agx::Real vx, agx::Real vy, agx::Real vz )
  {
    setAngularVelocity(agx::Vec3(vx,vy,vz));
  }

  inline void RigidBody::addForce( agx::Real fx, agx::Real fy, agx::Real fz )
  {
    addForce( agx::Vec3(fx,fy,fz) );
  }

  inline void RigidBody::addForceAtPosition( agx::Real fx, agx::Real fy, agx::Real fz, agx::Real px, agx::Real py, agx::Real pz )
  {
    addForceAtPosition( agx::Vec3(fx,fy,fz), agx::Vec3(px,py,pz) );
  }

  inline void RigidBody::addForceAtLocalPosition( agx::Real fx, agx::Real fy, agx::Real fz, agx::Real px, agx::Real py, agx::Real pz )
  {
    addForceAtLocalPosition( agx::Vec3(fx,fy,fz), agx::Vec3(px,py,pz) );
  }

  inline void RigidBody::addForceAtLocalCmPosition( agx::Real fx, agx::Real fy, agx::Real fz, agx::Real px, agx::Real py, agx::Real pz )
  {
    addForceAtLocalCmPosition( agx::Vec3(fx,fy,fz), agx::Vec3(px,py,pz) );
  }

  inline void RigidBody::addTorque( agx::Real tx, agx::Real ty, agx::Real tz )
  {
    addTorque( agx::Vec3(tx,ty,tz) );
  }

  inline void RigidBody::addLocalTorque( agx::Real tx, agx::Real ty, agx::Real tz )
  {
    addLocalTorque( agx::Vec3(tx, ty, tz) );
  }

  inline void RigidBody::setForce( agx::Real fx, agx::Real fy, agx::Real fz )
  {
    setForce( agx::Vec3(fx,fy,fz) );
  }

  inline void RigidBody::setTorque( agx::Real tx, agx::Real ty, agx::Real tz )
  {
    setTorque( agx::Vec3(tx,ty,tz) );
  }

  inline bool RigidBody::isPowerlineBody() const
  {
    return m_entity.state().powerline();
  }

  inline void RigidBody::setIsPowerlineBody(bool isHydraulicBody)
  {
    m_entity.state().setPowerline(isHydraulicBody);
  }

} // namespace agx

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
