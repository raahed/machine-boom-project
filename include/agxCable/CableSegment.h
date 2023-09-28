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

#include <agxSDK/LinkedSegment.h>

#include <agxCable/Attachment.h>
#include <agxCable/export.h>


namespace agxCollide
{
  class Geometry;
}


namespace agxCable
{
  class Cable;
}


namespace agxCable
{
  AGX_DECLARE_POINTER_TYPES(CableSegment);
  AGX_DECLARE_VECTOR_TYPES(CableSegment);

  /**
  A CableSegment represents a single link of a cable.
  */
  class AGXCABLE_EXPORT CableSegment : public agxSDK::LinkedSegment
  {
    public:
      /**
      Create a cable segment with its begin position at the given placement.

      \param transform - Position of the start of the segment and orientation.
      */
      CableSegment(const agx::AffineMatrix4x4& transform);

      /**
      The bend rotational force of the constraint linking this segment to the
      previous one. Always zero for the first segment since it doesn't have a
      previous segment.

      \return Sum of the rotational constraint forces along the two bend axes between this segment and the previous one.
      */
      agx::Real getBendTension();

      /**
      The twist rotational force of the constraint linking this segment to the
      previous one. Always zero for the first segment since it doesn't have a
      previous segment.

      \return The rotational constraint force along the twist axis between this segment and the previous one.
      */
      agx::Real getTwistTension();


      /**
      The translational force of the constraint linking this segment to
      the previous one. Always zero for the first segment since it doesn't have
      a previous segment.

      \return The translational force between this segment and the previous one.
      */
      agx::Real getStretchTension();

      /**
      \return The cable this segment is part of, if any, or nullptr.
      */
      agxCable::Cable* getCable() const;

      /**
      Get the collision geometry for this segment. Will return nullptr until the
      cable this segment is part of has been initialized.

      \return The collision geometry for this segment, if initialized, or nullptr.
      */
      agxCollide::Geometry* getGeometry() const;

      /**
      Attach this cable segment to a RigidBody using the given attachment.

      The recommended way of attaching bodies to a cable is to either create
      BodyFixedNodes during routing or using agxCable::Cable::attach.

      \see agxCable::BodyFixedNode
      \see agxCable::Cable::attach
      \param attachment
       \return
      */
      bool add(agxCable::SegmentAttachment* attachment);

      /**
      \return A list of all attachments this segment has.
      */
      const agxCable::SegmentAttachmentRefVector& getAttachments() const;

      /**
      Transfer all attachments added to this cable segment to the given segment.
      The transfer can only be performed if neither segment has already been
      made part of a cable.
      */
      bool giveAttachmentsTo(CableSegment* other);


    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      /**
      Constructor used for deserialization of legacy archives. Takes a
      parameter for each member that should be restored.
      */
      CableSegment(agx::RigidBody* body, const agx::Vec3& halfExtents, agx::Constraint* constraint);
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::CableSegment);
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      CableSegment();
      ~CableSegment();

    private:
      friend class Cable;
      void setHalfExtents(const agx::Vec3& halfExtents);
      void createGeometry(agx::Real length, agx::Real radius);

    private:
      agxCable::SegmentAttachmentRefVector m_attachments;
  };
}
