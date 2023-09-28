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

#include <agx/Referenced.h>
#include <agx/RigidBody.h>

#include <agx/Constraint.h>
#include <agx/BallJoint.h>
#include <agx/LockJoint.h>


#include <agx/AffineMatrix4x4.h>
#include <agx/Vec3.h>

#include <agxStream/Serializable.h>

#include <agxCable/export.h>



namespace agxCable
{
  class CableSegment;
}



namespace agxCable
{
  AGX_DECLARE_POINTER_TYPES(SegmentAttachment);
  AGX_DECLARE_VECTOR_TYPES(SegmentAttachment);


  class AGXCABLE_EXPORT SegmentAttachment : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Each SegmentAttachment implementation controls the initialization mode.
      */
      enum InitializationMode
      {
        USE_RELATIVE_TRANSFORM,
        ROTATE_TO_NODE_DIRECTION
      };


      template<typename T>
      static agxCable::SegmentAttachmentRef createFromWorld(const agx::Vec3& worldPoint, agx::RigidBody* body);

      template<typename T>
      static agxCable::SegmentAttachmentRef createFromBody(const agx::Vec3& bodyPoint, agx::RigidBody* body);

      template<typename T>
      static agxCable::SegmentAttachmentRef createFromWorld(
        const agx::AffineMatrix4x4& worldTransform, agx::RigidBody* body);

      template<typename T>
      static agxCable::SegmentAttachmentRef createFromBody(const agx::AffineMatrix4x4& bodyTransform, agx::RigidBody* body);


      void setName( const agx::Name& name );
      const agx::Name& getName() const;

      agx::Frame* getFrame() const;

      agx::Constraint* getConstraint() const;

      agx::RigidBody* getRigidBody() const;

      agx::AffineMatrix4x4 getWorldMatrix() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      virtual agx::Bool initialize(agxCable::CableSegment* segment);

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxCable::SegmentAttachment);
      void store(agxStream::OutputArchive& out) const override;
      void restore(agxStream::InputArchive& in) override;
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      SegmentAttachment(agx::RigidBody* body, agx::Frame* rbSegmentAttachmentFrame,
                        agxCable::SegmentAttachment::InitializationMode initMode);

      SegmentAttachment(const agx::Name& name, agx::RigidBody* body, agx::Frame* bodyAttachmentFrame,
                        agx::Constraint* constraint, InitializationMode initMode);

      virtual ~SegmentAttachment();

      virtual agx::ConstraintRef createConstraint(
        agx::RigidBodyRef body1, agx::FrameRef frame1,
        agx::RigidBodyRef body2, agx::FrameRef frame2) const = 0;


    protected:
      agx::Name m_name;
      agx::RigidBodyRef m_body;
      agx::FrameRef m_bodyAttachmentFrame;
      agx::ConstraintRef m_constraint;
      InitializationMode m_initMode;
  };




  /**
  Helper class to create give type of constraint for the different SegmentAttachments.
  */
  template<typename T>
  class SegmentAttachmentT : public agxCable::SegmentAttachment
  {
    public:
      /**
      Construct given rigid body and a relative frame.
      \param body - rigid body (null for world SegmentAttachment)
      \param rbSegmentAttachmentFrame - relative (to rigid body) frame
      \param initMode - The initialization mode, see InitializationMode
      */
      SegmentAttachmentT(
        agx::RigidBody* body, agx::Frame* rbSegmentAttachmentFrame,
        agxCable::SegmentAttachment::InitializationMode initMode)
        : SegmentAttachment(body, rbSegmentAttachmentFrame, initMode)
      {
      }

    protected:
      /**
      Default constructor for serialization.
      */
      SegmentAttachmentT()
          : SegmentAttachment(nullptr, nullptr, agxCable::SegmentAttachment::ROTATE_TO_NODE_DIRECTION)
      {
      }


      SegmentAttachmentT(const agx::Name& name, agx::RigidBody* body, agx::Frame* bodyAttachmentFrame,
                        agx::Constraint* constraint, agxCable::SegmentAttachment::InitializationMode initMode)
          : SegmentAttachment(name, body, bodyAttachmentFrame, constraint, initMode)
      {
      }

      /**
      Reference counted object, protected destructor.
      */
      virtual ~SegmentAttachmentT() {}

      /**
      \return new constraint of type T
      */
      virtual agx::ConstraintRef createConstraint(
        agx::RigidBodyRef body1, agx::FrameRef frame1,
        agx::RigidBodyRef body2, agx::FrameRef frame2) const override
      {
        return new T(body1, frame1, body2, frame2);
      }
  };

#ifdef SWIG
%template(SegmentAttachmentBallJoint) agxCable::SegmentAttachmentT<agx::BallJoint>;
%template(SegmentAttachmentLockJoint) agxCable::SegmentAttachmentT<agx::LockJoint>;
#endif

  class AGXCABLE_EXPORT PointSegmentAttachment : public agxCable::SegmentAttachmentT<agx::BallJoint>
  {
    public:
      PointSegmentAttachment(agx::RigidBody* body, const agx::Vec3& relativeTranslate);

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      PointSegmentAttachment(const agx::Name& name, agx::RigidBody* body, agx::Frame* bodyAttachmentFrame,
                             agx::Constraint* constraint, agxCable::SegmentAttachment::InitializationMode initMode)
          : SegmentAttachmentT<agx::BallJoint>(name, body, bodyAttachmentFrame, constraint, initMode)
      {
      }

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::PointSegmentAttachment);
      using agxCable::SegmentAttachment::restore;
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      PointSegmentAttachment();
      virtual ~PointSegmentAttachment();
  };



  class AGXCABLE_EXPORT RigidSegmentAttachment : public agxCable::SegmentAttachmentT<agx::LockJoint>
  {
    public:
      RigidSegmentAttachment(agx::RigidBody* body, const agx::Vec3& relativeTranslate);
      RigidSegmentAttachment(agx::RigidBody* body, const agx::AffineMatrix4x4& relativeTransform);


    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      RigidSegmentAttachment(const agx::Name& name, agx::RigidBody* body, agx::Frame* bodyAttachmentFrame,
                             agx::Constraint* constraint, agxCable::SegmentAttachment::InitializationMode initMode)
          : SegmentAttachmentT<agx::LockJoint>(name, body, bodyAttachmentFrame, constraint, initMode)
      {
      }

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::RigidSegmentAttachment);
      using agxCable::SegmentAttachment::restore;
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      RigidSegmentAttachment();
      virtual ~RigidSegmentAttachment();
  };



  template<typename T>
  agxCable::SegmentAttachmentRef agxCable::SegmentAttachment::createFromWorld(
    const agx::Vec3& worldPoint, agx::RigidBody* body)
  {
    agx::Vec3 localPoint = body != nullptr ? body->getFrame()->transformPointToLocal(worldPoint) : worldPoint;
    return createFromBody<T>(localPoint, body);
  }



  template<typename T>
  agxCable::SegmentAttachmentRef agxCable::SegmentAttachment::createFromBody(
    const agx::Vec3& bodyPoint, agx::RigidBody* body)
  {
    return new T(body, bodyPoint);
  }



  template<typename T>
  agxCable::SegmentAttachmentRef agxCable::SegmentAttachment::createFromWorld(
    const agx::AffineMatrix4x4& worldTransform, agx::RigidBody* body)
  {
    agx::AffineMatrix4x4 localTransform =
        body != nullptr ? worldTransform * body->getTransform().inverse() : worldTransform;
    return createFromBody<T>(localTransform, body);
  }



  template<typename T>
  agxCable::SegmentAttachmentRef agxCable::SegmentAttachment::createFromBody(
    const agx::AffineMatrix4x4& bodyTransform, agx::RigidBody* body)
  {
    return new T(body, bodyTransform);
  }
}
