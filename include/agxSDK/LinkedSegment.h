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

#include <agxSDK/LinkedStructureObjectData.h>

#include <agx/RigidBody.h>
#include <agx/Constraint.h>

#include <agxUtil/agxUtil.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( LinkedSegment );
  using LinkedSegmentContainer = agx::Vector<agxSDK::LinkedSegmentRef>;

  /**
  Segment object in a linked structure. This object has well defined extents
  that can be independent of the actual shape of it. The segment is by definition
  constrained with the segment before in the linked structure.

  The model center is by definition the begin/start point of this segment.
  */
  class AGXPHYSICS_EXPORT LinkedSegment : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Construct given transform and half extents of the shape. Note that a
      rigid body instance without a geometry/shape is created.
      \param transform - initial transform of this segment
      \param halfExtents - half extents of this segment, defining length, width
                           and thickness of the linked structure
      */
      LinkedSegment( const agx::AffineMatrix4x4& transform, const agx::Vec3& halfExtents );

      /**
      \return the rigid body of this segment
      */
      agx::RigidBody* getRigidBody() const;

      /**
      Returns the constraint created between this segment and the previous one in
      the linked structure. This constraint may be of any type and has to be created
      using createConstraintWithPrev.
      \sa createConstraintWithPrev
      \return the constraint constraining this segment with the previous segment in the linked structure
      */
      agx::Constraint* getConstraint() const;

      /**
      \return half extents of this segment
      */
      const agx::Vec3& getHalfExtents() const;

      /**
      \return the length of this segment
      */
      agx::Real getLength() const;

      /**
      \return The distance from the beginning of this segment to the beginning of the next one.
              Returns \p getLength if there is no next segment.
      */
      agx::Real getCurrentLength() const;

      /**
      \return the begin point of this segment in world coordinate frame
      (the begin point is where the constraint with the previous segment is attached)
      */
      agx::Vec3 getBeginPosition() const;

      /**
      \return the end point of this segment in world coordinate frame
      (the end point is where the constraint with the next segment is attached)
      */
      agx::Vec3 getEndPosition() const;

      /**
      \return the center position of this segment in world coordinate frame
      */
      agx::Vec3 getCenterPosition() const;

      /**
      \return the direction vector of this segment in world coordinate frame
      */
      agx::Vec3 getDirection() const;

      /**
      \return index in linked structure container, agx::InvalidIndex if not part of a linked structure
      */
      agx::UInt getIndex() const;

      /**
      \return the linked structure this segment is part of, otherwise null
      */
      agxSDK::LinkedStructure* getLinkedStructure() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::LinkedSegment );

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Create constraint with previous segment. The created constraint is stored
      in this object and accessed through getConstraint. It's not valid to
      call this method if getConstraint() != nullptr.
      */
      template<typename T>
      T* createConstraintWithPrev( const agxSDK::LinkedSegment& prev,
                                   const agx::Vec3& localConstraintAxis,
                                   const agx::Vec3& localConstraintY,
                                   agx::Bool disableCollisions );

      /**
      \param rb - rigid body instance
      \return linked segment for \p rb if \p rb belongs to a linked structure
      */
      static LinkedSegment* get( agx::RigidBody* rb );

      /**
      \param rb - rigid body instance
      \return linked segment for \p rb if \p rb belongs to a linked structure
      */
      static const LinkedSegment* get( const agx::RigidBody* rb );

      /**
      \param rb - rigid body instance
      \return segment of type for \p rb if \p rb belongs to a linked structure
      */
      template<typename T>
      static T* getT( agx::RigidBody* rb );

      /**
      \param rb - rigid body instance
      \return segment of type for \p rb if \p rb belongs to a linked structure
      */
      template<typename T>
      static const T* getT( const agx::RigidBody* rb );
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Default constructor used by serialization.
      */
      LinkedSegment();
      LinkedSegment(agx::RigidBody* body, const agx::Vec3& halfExtents, agx::Constraint* constraint);

      /**
      Reference counted object - protected destructor.
      */
      virtual ~LinkedSegment();

      void setHalfExtents(const agx::Vec3& halfExtens);
      void setRigidBody(agx::RigidBody* body);
      void setConstraint(agx::Constraint* constraint);

    private:
      friend class LinkedStructure;

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      \return the extra data this segment's rigid body contains
      */
      LinkedStructureObjectData* getData() const;

      /**
      Assign index to this segment in the linked structure.
      */
      void setIndex( agx::UInt index );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::UInt m_index;
      agx::RigidBodyRef m_rb;
      agx::Vec3 m_halfExtents;
      agx::ConstraintRef m_constraint;
  };

  inline agx::UInt LinkedSegment::getIndex() const
  {
    return m_index;
  }

  inline LinkedStructureObjectData* LinkedSegment::getData() const
  {
    return LinkedStructureObjectData::get<LinkedStructureObjectData>( m_rb );
  }

  inline LinkedStructure* LinkedSegment::getLinkedStructure() const
  {
    const auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( m_rb );
    return data != nullptr ? data->linkedStructure : nullptr;
  }

  template<typename T>
  T* LinkedSegment::createConstraintWithPrev( const LinkedSegment& prev,
                                              const agx::Vec3& localConstraintAxis,
                                              const agx::Vec3& localConstraintY,
                                              agx::Bool disableCollisions )
  {
    agxAssert( m_constraint == nullptr );
    if ( m_constraint != nullptr )
      return nullptr;

    const auto localConstraintX = localConstraintY ^ localConstraintAxis;
    agx::OrthoMatrix3x3 localRotation = agx::OrthoMatrix3x3( localConstraintX.x(), localConstraintX.y(), localConstraintX.z(),
                                                             localConstraintY.x(), localConstraintY.y(), localConstraintY.z(),
                                                             localConstraintAxis.x(), localConstraintAxis.y(), localConstraintAxis.z() );

    agx::FrameRef prevFrame = new agx::Frame();
    agx::FrameRef thisFrame = new agx::Frame();

    prevFrame->setLocalMatrix( agx::AffineMatrix4x4( localRotation,
                                                     agx::Vec3( 0, 0, prev.getLength() ) ) );
    thisFrame->setLocalMatrix( agx::AffineMatrix4x4( localRotation, agx::Vec3( 0, 0, 0 ) ) );

    m_constraint = new T( prev.getRigidBody(), prevFrame, getRigidBody(), thisFrame );

    if ( disableCollisions )
      agxUtil::setEnableCollisions( prev.getRigidBody(), getRigidBody(), false );

    return m_constraint->as<T>();
  }

  inline LinkedSegment* LinkedSegment::get( agx::RigidBody* rb )
  {
    auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( rb );
    if ( data == nullptr )
      return nullptr;
    return data->segment;
  }

  inline const LinkedSegment* LinkedSegment::get( const agx::RigidBody* rb )
  {
    const auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( rb );
    if ( data == nullptr )
      return nullptr;
    return data->segment;
  }

  template<typename T>
  T* LinkedSegment::getT( agx::RigidBody* rb )
  {
    auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( rb );
    if ( data == nullptr || data->segment == nullptr )
      return nullptr;
    return data->segment->template asSafe<T>();
  }

  template<typename T>
  const T* LinkedSegment::getT( const agx::RigidBody* rb )
  {
    const auto data = LinkedStructureObjectData::get<LinkedStructureObjectData>( rb );
    if ( data == nullptr || data->segment == nullptr )
      return nullptr;
    return data->segment->template asSafe<T>();
  }
}
