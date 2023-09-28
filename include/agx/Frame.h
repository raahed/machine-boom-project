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

#ifndef AGX_FRAME_H
#define AGX_FRAME_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable: 4290) //: C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

#include <agx/observer_ptr.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/OrthoMatrix3x3.h>

#include <agxStream/Serializable.h>
#include <agx/SetVector.h>
#include <agxData/EntityPtr.h>
#include <agxData/AttributePtr.h>
#include <agx/Physics/FrameEntity.h>

namespace agxSDK
{
  class Assembly;
}

namespace agxCollide
{
  class Geometry;
}

namespace agx
{
  class RigidBody;
  class EulerAngles;
  class OrthoMatrix3x3;

  AGX_DECLARE_POINTER_TYPES(Frame);

  /**
  The object defining a frame of reference and providing transformations operations.

  A frame can have a parent. If so, localToWorld transformation is the concatenation
  of this frames transformation with the parents.

  A frame can convert vectors and points into its parents coordinate system.

  A frame of reference can be kinematic in which case the
  velocity and angular velocities are important objects.

  A frame can internally be set to static, which means that getVelocity will always return zero.
  */
  class AGXPHYSICS_EXPORT Frame : public Referenced, public agxStream::Serializable
  {
    public:
      /**
      Default constructor.
      */
      Frame();

      /**
      Frame with existing matrix storage.
      */
      Frame( const agxData::AttributePtr< AffineMatrix4x4 >& localTransform, const agxData::AttributePtr< AffineMatrix4x4 >& globalTransform );

      /**
      Instantiate a frame with relative transformation matrix.
      \param matrix - transformation matrix that will be this frame's local matrix
      */
      explicit Frame( const agx::AffineMatrix4x4 matrix );

      /**
      Copy constructor.
      \param frame - other frame
      */
      Frame( const agx::Frame& frame );

      /**
      Assignment operator.
      \param frame - other frame
      */
      Frame& operator = ( const agx::Frame& frame );

    public:
      /**
      Set the parent of this Frame.

      This means that getTranslate, getRotate, getTransform will be given
      in the parents coordinate frame. I.e., this frame's transformation
      will be concatenated with the parents.
      \param frame - new parent frame, 0 to remove parent
      \return true if parent is changed (not the same as before) - otherwise false
      */
      bool setParent( agx::Frame* frame );

      /**
      \return a pointer to the parent frame, 0 if this frame has no parent
      */
      agx::Frame* getParent();

      /**
      \return a pointer to the parent frame, 0 if this frame has no parent
      */
      const agx::Frame* getParent() const;

      /**
      \return true if this frame has a parent - otherwise false
      */
      bool hasParent() const;

      /**
      This method will set the local matrix of this frame to be the matrix \p m
      multiplied with the inverse of the parents matrix:
        M = m * inv(parent)
      I.e., assign the final (world) transform of this frame.
      \param matrix - final transform for this frame (leads to local transform = matrix * inv(parent))
      */
      void setMatrix( const agx::AffineMatrix4x4& matrix );

      /**
      \return the transformation matrix for this frame including parents coordinate frames
              (this frame's final (world) transform)
      */
      const agx::AffineMatrix4x4& getMatrix() const;


      /**
      \return true iff all parts of this frame are finite values (no inf:s, no NaN:s)
      */
      bool isFinite() const;

      /**
      Assign the local transformation matrix for this frame ignoring any eventual parent transformation.
      \param matrix - relative to parent transform for this frame
      */
      void setLocalMatrix( const agx::AffineMatrix4x4& matrix );

      /**
      \return the local transformation matrix (i.e., the transformation relative to the parent)
      */
      const agx::AffineMatrix4x4& getLocalMatrix() const;

      /**
      \return the translate of this frame in final (world) coordinate frame
      */
      agx::Vec3 getTranslate() const;

      /**
      Assign the final (world) translate of this frame.
      \param translate - final (world) translate of this frame
      */
      void setTranslate( const agx::Vec3& translate );

      /**
      Assign the final (world) translate of this frame.
      \param x - final (world) x translate of this frame
      \param y - final (world) y translate of this frame
      \param z - final (world) z translate of this frame
      */
      void setTranslate( agx::Real x, agx::Real y, agx::Real z );

      /**
      \return the relative translate to this frame's parent frame
      */
      agx::Vec3 getLocalTranslate() const;

      /**
      Assign the parent relative translate of this frame.
      \param translate - parent relative translate
      */
      void setLocalTranslate( const agx::Vec3& translate );

      /**
      Assign the parent relative translate of this frame.
      \param x - parent relative x translate
      \param y - parent relative y translate
      \param z - parent relative z translate
      */
      void setLocalTranslate( agx::Real x, agx::Real y, agx::Real z );

      /**
      \return the rotation of this frame in final (world) coordinate frame
      */
      agx::Quat getRotate() const;

      /**
      Assign the final (world) rotation of this frame.
      \param rotation - final (world) rotate
      */
      void setRotate( const agx::Quat& rotation );

      /**
      Assign the final (world) rotation of this frame.
      \param rotation - final (world) rotate
      */
      void setRotate( const agx::OrthoMatrix3x3& rotation );

      /**
      Assign the final (world) rotation of this frame.
      \param euler - final (world) rotate
      */
      void setRotate( const agx::EulerAngles& euler);

      /**
      \return the relative rotation to this frame's parent frame
      */
      Quat getLocalRotate() const;

      /**
      Assign the parent relative rotation of this frame.
      \param rotation - parent relative rotation
      */
      void setLocalRotate( const agx::Quat& rotation );

      /**
      Assign the parent relative rotation of this frame.
      \param rotation - parent relative rotation
      */
      void setLocalRotate( const agx::EulerAngles& rotation );

      /**
      Transform vector from this frame to the world frame.
      \param vectorLocal - vector to transform, given in this frame
      \return the given vector transformed to the world coordinate frame
      */
      agx::Vec3 transformVectorToWorld( const agx::Vec3& vectorLocal ) const;

      /**
      Transform vector from this frame to the world frame.
      \param vx - x-component to transform, given in this frame
      \param vy - y-component to transform, given in this frame
      \param vz - z-component to transform, given in this frame
      \return the given vector transformed to the world coordinate frame
      */
      agx::Vec3 transformVectorToWorld( agx::Real vx, agx::Real vy, agx::Real vz ) const;

      /**
      Transform vector from this frame to the world frame.
      \param localVector - vector to transform, given in this frame
      \param worldVector - the transformed vector given in the world coordinate frame
      \return a reference to the transformed \p worldVector vector
      */
      agx::Vec3& transformVectorToWorld( const agx::Vec3& localVector, agx::Vec3& worldVector ) const;

      /**
      Transform point from this frame to the world frame.
      \param pointLocal - point to transform, given in this frame
      \return the given point transformed to the world coordinate frame
      */
      agx::Vec3 transformPointToWorld( const agx::Vec3& pointLocal ) const;

      /**
      Transform point from this frame to the world frame.
      \param px - x-component to transform, given in this frame
      \param py - y-component to transform, given in this frame
      \param pz - z-component to transform, given in this frame
      \return the given point transformed to the world coordinate frame
      */
      agx::Vec3 transformPointToWorld( agx::Real px, agx::Real py, agx::Real pz ) const;

      /**
      Transform point from this frame to the world frame.
      \param localPoint - point to transform, given in this frame
      \param worldPoint - the transformed point given in the world coordinate frame
      \return a reference to the transformed \p worldPoint point
      */
      agx::Vec3& transformPointToWorld( const agx::Vec3& localPoint, agx::Vec3& worldPoint ) const;

      /**
      Transform vector from the world coordinate frame to this frame.
      \param worldVector - vector to transform, given in the world coordinate world frame
      \return the given vector transformed to this frame
      */
      agx::Vec3 transformVectorToLocal( const agx::Vec3& worldVector ) const;

      /**
      Transform vector from the world coordinate frame to this frame.
      \param vx - x-component to transform, given in the world coordinate world frame
      \param vy - y-component to transform, given in the world coordinate world frame
      \param vz - z-component to transform, given in the world coordinate world frame
      \return the given vector transformed to this frame
      */
      agx::Vec3 transformVectorToLocal( agx::Real vx, agx::Real vy, agx::Real vz ) const;

      /**
      Transform vector from the world coordinate frame to this frame.
      \param vectorWorld - vector to transform, given in the world coordinate frame
      \param vectorLocal - transformed vector given in this frame
      \return a reference to the given world vector transformed to this frame
      */
      agx::Vec3& transformVectorToLocal( const agx::Vec3& vectorWorld, agx::Vec3& vectorLocal ) const;

      /**
      Transform point from the world coordinate frame to this frame.
      \param pointWorld - point to transform, given in the world coordinate frame
      \return the given point transformed to this frame
      */
      agx::Vec3 transformPointToLocal( const agx::Vec3& pointWorld ) const;

      /**
      Transform point from the world coordinate frame to this frame.
      \param px - x-component to transform, given in the world coordinate frame
      \param py - y-component to transform, given in the world coordinate frame
      \param pz - z-component to transform, given in the world coordinate frame
      \return the given point transformed to this frame
      */
      agx::Vec3 transformPointToLocal( agx::Real px, agx::Real py, agx::Real pz ) const;

      /**
      Transform point from the world coordinate frame to this frame.
      \param pointWorld - point to transform, given in the world coordinate frame
      \param pointLocal - transformed point given in this frame
      \return a reference to the given world point transformed to this frame
      */
      agx::Vec3& transformPointToLocal( const agx::Vec3& pointWorld, agx::Vec3& pointLocal ) const;

      /**
      This method takes the transformation stored in this frame, and concatenates it
      to all of its children. The transform for all frames between this and the leaf will be set to identity.
      So you can safely call transferDataIntoLeaves on a parent frame to a rigid body, and then
      remove the rigid body as a child of this frame. The transformation of the rigid body will be the same.

      The local velocity of bodies will be retained. So If you have a body moving along
      positive Y, add parent frame which rotates 90deg negative around Y, the body keep
      moving along positive Y (in World coordinates). But if you call
      parent->transferDataIntoLeaves( true ) the body will keep its local velocity,
      which means that it will start moving along positive X.

      \param keepLocalVelocity - if true, any bodies (which are leaves) will keep their
                                 local velocity (velocity will be transformed). If not,
                                 they will keep their current velocity (in world coordinates),
                                 i.e., velocities will NOT be transformed.
      \param clearTransform - If true, then the transformation of this local frame will be set to identity
                              at the end of this method. Otherwise this has to be done explicitly.
      */
      void transferDataIntoLeaves( bool keepLocalVelocity = true, bool clearTransform=true );

      /**
      Specifies whether a transferDataIntoLeaves() operation will go into this frames
      children (including any attached bodies) and concatenating transformations.
      Frames are usually setup in a hierarchy, a root->transferDataIntoLeaves() will
      take the transformation from the root concatenate transformation on the way
      down to the leave, where the transformation will be stored, in a body, or a
      leaf frame. If you call child->setAllowTransferDataIntoChildren( false ); the
      concatenated transformation will stop at the child, leaving its children untouched.
      \param allow - true if transfer is allowed
      */
      void setAllowTransferIntoChildren( bool allow );

      /**
      \return true if transferDataIntoLeaves() is allowed to traverse below this frame
      */
      bool allowTransferDataIntoChildren() const;

      /**
      Move all the frame children of this frame to the \p new parent while retaining
      the original global transformation. That is, the matrix (getMatrix()) of all
      children will be the same after the move. Can Throw exception if \p newFrame
      is child of this frame, because we cannot have frames being parents to them self.
      \param newParent - parent frame
      */
      void transferChildren( agx::Frame* newParent );

      /**
      \return true if this frame has children
      */
      bool hasChildren() const;

      /**
      Remove a specified frame from the list of children.
      \param frame - the child frame to remove
      \return true if removal was successful, false if frame wasn't found
      */
      bool removeChild( agx::Frame* frame );

      /**
      Remove all children of this frame.
      */
      void removeAllChildren();

      /**
      \return the rigid body that is direct associated to this frame, 0 if none
      */
      agx::RigidBody* getRigidBody();

      /**
      \return the rigid body that is direct associated to this frame, 0 if none
      */
      const agx::RigidBody* getRigidBody() const;

      /**
      Internal method.
      Assign transforms to this frame with existing matrix storage.
      */
      void setMatrixPointers( const agxData::AttributePtr< agx::AffineMatrix4x4 >& localTransform, const agxData::AttributePtr< agx::AffineMatrix4x4 >& globalTransform );

      /**
      Assign entity data.
      */
      void setEntityData(agxData::AttributePtr< agx::AffineMatrix4x4 > localTransform, agxData::AttributePtr< agx::AffineMatrix4x4 > globalTransform);

      /**
      Internal method.
      Updates local transform given global transform and parents.
      */
      void updateLocalMatrix();

      /**
      Internal method.
      Performs the same operation as updateLocalMatrix but will use SSE implementation instead.
      It is important that the data is aligned correctly and it is the callers responsibility to
      know if this method can be used.

      If in doubt, pick updateLocalMatrix instead.
      */
      void updateLocalMatrixSSE();

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Frame );

    protected:
      /**
      Protected destructor - reference counted object.
      */
      virtual ~Frame();

    private:
      friend class RigidBody;
      friend class agxCollide::Geometry;

      /**
      Assign rigid body that has this frame as primary frame.
      */
      void setRigidBody( agx::RigidBody* body );

      /**
      Assign geometry that has this frame as primary frame.
      */
      void setGeometry( agxCollide::Geometry* geometry );

      /**
      Propagate transform to all associated objects.
      */
      void propagateUpdate();

      /**
      Called when the transform has been updated. Will execute propagateUpdate
      and if this frame is associated to a rigid body, mass properties will
      also be updated.
      */
      void postTransformUpdate();

      /**
      Make \p frame a child of this frame.
      \param frame - frame to be added as a child to this frame
      \return true if child was added (was not already a child)
      */
      bool addChild( agx::Frame* frame );



    private:
      typedef SetVector< Frame* > FramePtrVector;

    private:
      Physics::FrameRef                                 m_entity;
      mutable agxData::AttributePtr< AffineMatrix4x4 >  m_globalTransform;
      agxData::AttributePtr<AffineMatrix4x4>            m_localTransform;

      agx::observer_ptr< agx::RigidBody >               m_rigidBody;
      agx::observer_ptr< agxCollide::Geometry >         m_geometry;
      bool                                              m_allowTransferDataIntoChildren;
      Frame*                                            m_parent;
      FramePtrVector                                    m_children;
  };

  AGX_FORCE_INLINE void Frame::setMatrix( const AffineMatrix4x4& matrix )
  {
    m_localTransform.get() = matrix;
    const Frame* parent = getParent();
    if (parent)
      m_localTransform.get().postMult(parent->getMatrix().inverse());

    this->postTransformUpdate();
  }

  AGX_FORCE_INLINE void Frame::updateLocalMatrix()
  {
    m_localTransform.get() = m_parent ? m_globalTransform.get() * m_parent->m_globalTransform.get().inverse() : m_globalTransform.get();
  }


  AGX_FORCE_INLINE Frame* Frame::getParent()
  {
    return m_parent;
  }

  AGX_FORCE_INLINE const Frame* Frame::getParent() const
  {
    return m_parent;
  }

  AGX_FORCE_INLINE bool Frame::hasParent() const
  {
    return m_parent != nullptr;
  }

  AGX_FORCE_INLINE const AffineMatrix4x4& Frame::getLocalMatrix() const
  {
    return m_localTransform.get();
  }

  AGX_FORCE_INLINE void Frame::setLocalMatrix( const AffineMatrix4x4& matrix )
  {
    m_localTransform.get() = matrix;
    this->postTransformUpdate();
  }

  AGX_FORCE_INLINE Vec3 Frame::getTranslate() const
  {
    return getMatrix().getTranslate();
  }

  AGX_FORCE_INLINE void Frame::setTranslate( const Vec3& translate )
  {
    if ( getParent() )
      m_localTransform.get().setTranslate ( translate * m_parent->getMatrix().inverse() );
    else
      m_localTransform.get().setTranslate( translate );

    this->postTransformUpdate();
  }

  AGX_FORCE_INLINE void Frame::setTranslate( Real x, Real y, Real z )
  {
    setTranslate( Vec3( x, y, z ) );
  }

  AGX_FORCE_INLINE Vec3 Frame::getLocalTranslate() const
  {
    return m_localTransform.get().getTranslate();
  }

  AGX_FORCE_INLINE void Frame::setLocalTranslate( const Vec3& translate )
  {
    m_localTransform.get().setTranslate( translate );
    this->postTransformUpdate();
  }

  AGX_FORCE_INLINE void Frame::setLocalTranslate( Real x, Real y, Real z )
  {
    setLocalTranslate( Vec3( x, y, z ) );
  }

  AGX_FORCE_INLINE Vec3 Frame::transformVectorToWorld( const Vec3& vectorLocal ) const
  {
    Vec3 vectorWorld;
    return transformVectorToWorld( vectorLocal, vectorWorld );
  }

  AGX_FORCE_INLINE Vec3 Frame::transformVectorToWorld( Real vx, Real vy, Real vz ) const
  {
    return transformVectorToWorld( Vec3( vx, vy, vz ) );
  }

  AGX_FORCE_INLINE Vec3& Frame::transformVectorToWorld( const Vec3& localVector, Vec3& worldVector ) const
  {
    const AffineMatrix4x4& m = getMatrix();

    m.transform3x3( localVector, worldVector );
    return worldVector;
  }

  AGX_FORCE_INLINE Vec3 Frame::transformPointToWorld( const Vec3& pointLocal ) const
  {
    Vec3 wP;
    return transformPointToWorld( pointLocal, wP);
  }

  AGX_FORCE_INLINE Vec3 Frame::transformPointToWorld( Real px, Real py, Real pz ) const
  {
    return transformPointToWorld( Vec3( px, py, pz ) );
  }

  AGX_FORCE_INLINE Vec3& Frame::transformPointToWorld( const Vec3& localPoint, Vec3& worldPoint ) const
  {
    const AffineMatrix4x4& m = getMatrix();

    worldPoint = m.preMult( localPoint );
    return worldPoint;
  }

  AGX_FORCE_INLINE Vec3 Frame::transformVectorToLocal( const Vec3& worldVector ) const
  {
    Vec3 localVector;
    return transformVectorToLocal( worldVector, localVector );
  }

  AGX_FORCE_INLINE Vec3 Frame::transformVectorToLocal( Real vx, Real vy, Real vz ) const
  {
    return transformVectorToLocal( Vec3( vx, vy, vz ) );
  }

  AGX_FORCE_INLINE Vec3& Frame::transformVectorToLocal( const Vec3& vectorWorld, Vec3& vectorLocal ) const
  {
    AffineMatrix4x4 m = getMatrix();

    m.setTranslate( 0, 0, 0 );
    vectorLocal = m.postMult( vectorWorld );
    return vectorLocal;
  }

  AGX_FORCE_INLINE Vec3 Frame::transformPointToLocal( const Vec3& pointWorld ) const
  {
    Vec3 pointLocal;
    return transformPointToLocal( pointWorld, pointLocal );
  }

  AGX_FORCE_INLINE Vec3 Frame::transformPointToLocal( Real px, Real py, Real pz ) const
  {
    return transformPointToLocal( Vec3( px, py, pz ) );
  }

  AGX_FORCE_INLINE Vec3& Frame::transformPointToLocal( const Vec3& pointWorld, Vec3& pointLocal ) const
  {
    AffineMatrix4x4 m = getMatrix().inverse();

    pointLocal = m.preMult( pointWorld );
    return pointLocal;
  }

  AGX_FORCE_INLINE bool Frame::allowTransferDataIntoChildren() const
  {
    return m_allowTransferDataIntoChildren;
  }

  AGX_FORCE_INLINE void Frame::setAllowTransferIntoChildren( bool allow )
  {
    m_allowTransferDataIntoChildren = allow;
  }

  AGX_FORCE_INLINE bool Frame::hasChildren() const
  {
    return !m_children.empty();
  }

  AGX_FORCE_INLINE void Frame::removeAllChildren()
  {
    FramePtrVector::iterator it = m_children.begin();
    for(; it != m_children.end(); ++it)
    {
      (*it)->m_parent = nullptr;
    }
    m_children.clear();
  }

  AGX_FORCE_INLINE bool Frame::removeChild( Frame* frame )
  {
    if(m_children.erase( frame ))
    {
      frame->m_parent = nullptr;
      return true;
    }
    return false;
  }

  AGX_FORCE_INLINE agx::RigidBody* Frame::getRigidBody()
  {
    return m_rigidBody.get();
  }

  AGX_FORCE_INLINE const agx::RigidBody* Frame::getRigidBody() const
  {
    return m_rigidBody;
  }

 #define AGX_ADD_FRAME_TRANSFORM_INTERFACE()                                                               \
   AGX_FORCE_INLINE const agx::AffineMatrix4x4& getTransform() const { return getFrame()->getMatrix(); }   \
   AGX_FORCE_INLINE agx::Vec3                   getPosition() const { return getFrame()->getTranslate(); } \
   AGX_FORCE_INLINE agx::Quat                   getRotation() const { return getFrame()->getRotate(); }

} // namespace agx

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
