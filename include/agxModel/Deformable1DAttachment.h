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

#ifndef AGXMODEL_DEFORMABLE1DATTACHMENT_H
#define AGXMODEL_DEFORMABLE1DATTACHMENT_H

#include <agxModel/export.h>

#include <agx/LockJoint.h>
#include <agx/BallJoint.h>
#include <agx/Hinge.h>
#include <agx/Prismatic.h>

namespace agxModel
{
  class Deformable1DNode;

  AGX_DECLARE_POINTER_TYPES( NodeAttachment );

  /**
  Base class for attachments to Deformable1D nodes.
  */
  class AGXMODEL_EXPORT NodeAttachment : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Each attachment implementation controls the initialization mode.
      */
      enum InitializationMode
      {
        USE_RELATIVE_TRANSFORM,
        ROTATE_TO_NODE_DIRECTION
      };

    public:
      /**
      Creates an node attachment given a point in world frame. The attachment frame
      will coincide at the given world point.
      \param worldPoint - constraint center point given in the world frame
      \param rb         - rigid body attached to
      \return the attachment of type T
      */
      template< typename T >
      static agxModel::NodeAttachmentRef createFromWorld( agx::Vec3 worldPoint, agx::RigidBody* rb );

      /**
      Creates an node attachment given a point in \p rb local frame. The attachment frame
      will coincide at the given local point.
      \param bodyPoint - constraint center point given in the local body frame
      \param rb        - rigid body attached to
      \return the attachment of type T
      */
      template< typename T >
      static agxModel::NodeAttachmentRef createFromBody( agx::Vec3 bodyPoint, agx::RigidBody* rb );

      /**
      Creates an node attachment given a transform in world frame. The attachment frame
      will coincide at the given world frame.
      \param worldTransform - constraint transform given in the world frame
      \param rb             - rigid body attached to
      \return the attachment of type T
      */
      template< typename T >
      static agxModel::NodeAttachmentRef createFromWorld( agx::AffineMatrix4x4 worldTransform, agx::RigidBody* rb );

      /**
      Creates an node attachment given a transform in \p rb local frame. The attachment frame
      will coincide at the given local transform.
      \param bodyTransform - constraint transform given in the local body frame
      \param rb            - rigid body attached to
      \return the attachment of type T
      */
      template< typename T >
      static agxModel::NodeAttachmentRef createFromBody( agx::AffineMatrix4x4 bodyTransform, agx::RigidBody* rb );

    public:
      /**
      Name this attachment, e.g., to find this attachment using agxModel::Node::getAttachment( const agx::String& name ).
      \param name - name of this attachment
      */
      void setName( const agx::String& name );

      /**
      \return name of this attachment
      */
      const agx::String& getName() const;

      /**
      \return relative frame
      */
      agx::Frame* getFrame() const;

      /**
      \return the constraint of this attachment
      */
      agx::Constraint* getConstraint() const;

      /**
      \return the rigid body
      */
      agx::RigidBody* getRigidBody() const;

      /**
      \return world transform of this attachment
      */
      agx::AffineMatrix4x4 getWorldMatrix() const;

      InitializationMode getInitMode() const;


    public:
      /**
      Initializes constraint.
      */
      virtual agx::Bool initialize(agxModel::Deformable1DNode* node);

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxModel::NodeAttachment );

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Construct given rigid body and a relative frame.
      \param rb - rigid body (null for world attachment)
      \param rbAttachmentFrame - relative (to rigid body) frame
      \param initMode - The mode for the attachment see ::InitializationMode
      */
      NodeAttachment( agx::RigidBody* rb, agx::Frame* rbAttachmentFrame, agxModel::NodeAttachment::InitializationMode initMode );

      /**
      Serialization store.
      */
      void store( agxStream::OutputArchive& out ) const override;

      /**
      Serialization restore.
      */
      void restore( agxStream::InputArchive& in ) override;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~NodeAttachment();

      /**
      Create the attachment constraint. Default: Lock Joint.
      */
      virtual agx::ConstraintRef createConstraint( agx::RigidBodyRef rb1, agx::FrameRef f1, agx::RigidBodyRef rb2, agx::FrameRef f2 ) const = 0;

    protected:
      agx::String        m_name;
      agx::RigidBodyRef  m_rb;
      agx::FrameRef      m_rbAttachmentFrame;
      agx::ConstraintRef m_constraint;
      InitializationMode m_initMode;
  };

  /**
  Helper class to create give type of constraint for the different attachments.
  */
  template< typename T >
  class NodeAttachmentT : public agxModel::NodeAttachment
  {
    public:
      /**
      Construct given rigid body and a relative frame.
      \param rb - rigid body (null for world attachment)
      \param rbAttachmentFrame - relative (to rigid body) frame
      \param initMode - The initialization mode, see ::InitializationMode
      */
      NodeAttachmentT( agx::RigidBody* rb, agx::Frame* rbAttachmentFrame, agxModel::NodeAttachment::InitializationMode initMode ) : NodeAttachment( rb, rbAttachmentFrame, initMode ) {}

    protected:
      /**
      Default constructor for serialization.
      */
      NodeAttachmentT() : NodeAttachment( nullptr, nullptr, agxModel::NodeAttachment::ROTATE_TO_NODE_DIRECTION ) {}

      /**
      Reference counted object, protected destructor.
      */
      virtual ~NodeAttachmentT() {}

      /**
      \return new constraint of type T
      */
      virtual agx::ConstraintRef createConstraint( agx::RigidBodyRef rb1, agx::FrameRef f1, agx::RigidBodyRef rb2, agx::FrameRef f2 ) const override
      {
        return new T( rb1, f1, rb2, f2 );
      }
  };

#ifdef SWIG
  %template(NodeAttachmentTBallJoint) agxModel::NodeAttachmentT<agx::BallJoint>;
  %template(NodeAttachmentTLockJoint) agxModel::NodeAttachmentT<agx::LockJoint>;
  %template(NodeAttachmentTHinge)     agxModel::NodeAttachmentT<agx::Hinge>;
  %template(NodeAttachmentTPrismatic) agxModel::NodeAttachmentT<agx::Prismatic>;
#endif

  /**
  Node attachment using ball joint, i.e., the node will rotate about this attachment.
  */
  class AGXMODEL_EXPORT PointNodeAttachment : public agxModel::NodeAttachmentT< agx::BallJoint >
  {
    public:
      /**
      Construct given rigid body and relative (to rigid body) position.
      \param rb - rigid body
      \param relPosition - position in rigid body frame
      */
      PointNodeAttachment( agx::RigidBody* rb, const agx::Vec3& relPosition );

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::PointNodeAttachment );

    protected:
      /**
      Default constructor for serialization.
      */
      PointNodeAttachment();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~PointNodeAttachment();
  };

  /**
  Node attachment using lock joint, i.e., the node cannot move relative to this attachment.
  */
  class AGXMODEL_EXPORT RigidNodeAttachment : public agxModel::NodeAttachmentT< agx::LockJoint >
  {
    public:
      /**
      Construct given rigid body and relative (to rigid body) position. The direction
      of the Deformable1D will be used.
      \param rb - rigid body
      \param relPosition - position in rigid body frame
      */
      RigidNodeAttachment( agx::RigidBody* rb, const agx::Vec3& relPosition );

      /**
      Construct given rigid body and relative (to rigid body) transform.
      The relative transform defines the lock joint.
      \param rb - rigid body
      \param relTransform - transform in rigid body frame
      */
      RigidNodeAttachment( agx::RigidBody* rb, const agx::AffineMatrix4x4& relTransform );

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::RigidNodeAttachment );

    protected:
      /**
      Default constructor for serialization.
      */
      RigidNodeAttachment();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~RigidNodeAttachment();
  };

  /**
  Node attachment using hinge joint, i.e., the node can rotate about this attachment.
  */
  class AGXMODEL_EXPORT RotatingNodeAttachment : public agxModel::NodeAttachmentT< agx::Hinge >
  {
    public:
      /**
      Construct given rigid body and relative position. The hinge axis will be calculated
      to be along the Deformable1D.
      \param rb - rigid body
      \param relPosition - position in rigid body frame
      */
      RotatingNodeAttachment( agx::RigidBody* rb, const agx::Vec3& relPosition );

      /**
      Construct given rigid body and relative transform. The relative transform defines the hinge.
      \param rb - rigid body
      \param relTransform - transform in rigid body frame
      */
      RotatingNodeAttachment( agx::RigidBody* rb, const agx::AffineMatrix4x4& relTransform );

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::RotatingNodeAttachment );

    protected:
      /**
      Default constructor for serialization.
      */
      RotatingNodeAttachment();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~RotatingNodeAttachment();
  };

  /**
  Node attachment using prismatic joint, i.e., the node can slide along this attachment.
  */
  class AGXMODEL_EXPORT SlidingNodeAttachment : public agxModel::NodeAttachmentT< agx::Prismatic >
  {
    public:
      /**
      Construct given rigid body and relative position. The prismatic axis will be calculated
      to be along the Deformable1D.
      \param rb - rigid body
      \param relPosition - position in rigid body frame
      */
      SlidingNodeAttachment( agx::RigidBody* rb, const agx::Vec3& relPosition );

      /**
      Construct given rigid body and relative transform. The relative transform defines the prismatic.
      \param rb - rigid body
      \param relTransform - transform in rigid body frame
      */
      SlidingNodeAttachment( agx::RigidBody* rb, const agx::AffineMatrix4x4& relTransform );

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::SlidingNodeAttachment );

    protected:
      /**
      Default constructor for serialization.
      */
      SlidingNodeAttachment();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~SlidingNodeAttachment();
  };

  template< typename T >
  agxModel::NodeAttachmentRef NodeAttachment::createFromWorld( agx::Vec3 worldPoint, agx::RigidBody* rb )
  {
    return createFromBody< T >( (rb != nullptr ? rb->getFrame()->transformPointToLocal( worldPoint ) : worldPoint), rb );
  }

  template< typename T >
  agxModel::NodeAttachmentRef NodeAttachment::createFromBody( agx::Vec3 bodyPoint, agx::RigidBody* rb )
  {
    return new T( rb, bodyPoint );
  }

  template< typename T >
  agxModel::NodeAttachmentRef NodeAttachment::createFromWorld( agx::AffineMatrix4x4 worldTransform, agx::RigidBody* rb )
  {
    return createFromBody< T >( (rb != nullptr ? worldTransform * rb->getTransform().inverse() : worldTransform), rb );
  }

  template< typename T >
  agxModel::NodeAttachmentRef NodeAttachment::createFromBody( agx::AffineMatrix4x4 bodyTransform, agx::RigidBody* rb )
  {
    return new T( rb, bodyTransform );
  }
}

#endif // AGXMODEL_DEFORMABLE1DATTACHMENT_H
