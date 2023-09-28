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

#ifndef AGXCABLE_NODE_H
#define AGXCABLE_NODE_H


#include <agxCable/CableSegment.h>
#include <agxCable/Attachment.h>

#include <agxStream/Serializable.h>


namespace agxCable
{

  AGX_DECLARE_POINTER_TYPES(RoutingNode);
  AGX_DECLARE_VECTOR_TYPES(RoutingNode);

  /**
  Base class for route nodes that defines the path and some behavior of a cable.
  */
  class AGXCABLE_EXPORT RoutingNode : public agxCable::CableSegment
  {
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::RoutingNode);
      using agxCable::CableSegment::restore;

    protected:
      DOXYGEN_START_INTERNAL_BLOCK()
      RoutingNode();
      RoutingNode(const agx::AffineMatrix4x4& transform);
      RoutingNode(agx::RigidBody* body, const agx::Vec3& halfExtents, agx::Constraint* constraint);
      virtual ~RoutingNode();
      DOXYGEN_END_INTERNAL_BLOCK()
  };



  AGX_DECLARE_POINTER_TYPES( BodyFixedNode );

  /**
  A cable node that is attached to a RigidBody. The cable segment created for
  this node will become locked to the body.
  */
  class AGXCABLE_EXPORT BodyFixedNode : public RoutingNode
  {
    public:
      /**
      Create a new BodyFixedNode attached to the given RigidBody. The relative
      translation is given in the body's model coordinate frame.

      The cable is attached so that it passes through the attachment point along
      the Z axis of the body's model coordinate frame. Use the constructor taking
      an AffineMatrix4x4 instead if another direction is required.

      \param body - The RigidBody that the cable should be attached to.
      \param relativeTranslate
      */
      BodyFixedNode( agx::RigidBody* body, const agx::Vec3& relativeTranslate );


      /**
      Create a new BodyFixedNode attached to the given RigidBody. The relative
      transform is the transformation from the body's model coordinate frame to
      the point through which the cable should pass. The cable will pass through
      this point along the transformed Z axis.
      */
      BodyFixedNode(agx::RigidBody* body, const agx::AffineMatrix4x4& relativeTransform);

      DOXYGEN_START_INTERNAL_BLOCK()
    public:
      BodyFixedNode(agx::RigidBody* body, const agx::Vec3& halfExtents, agx::Constraint* constraint);
      AGXSTREAM_DECLARE_SERIALIZABLE_CUSTOM_CREATE(agxCable::BodyFixedNode);

    protected:
      BodyFixedNode();
      virtual ~BodyFixedNode();

      DOXYGEN_END_INTERNAL_BLOCK()
  };



  AGX_DECLARE_POINTER_TYPES(FreeNode);

  /**
  A cable not that is simply a point in space. The cable will be routed through
  this node but is after that free to move in any way.
  */
  class AGXCABLE_EXPORT FreeNode : public RoutingNode
  {
    public:
      /**
      Create a new FreeNode at the given position, in world coordinates.
      */
      FreeNode(const agx::Vec3& worldPosition);

      /**
      Create a new FreeNode at the given position, in world coordinates.
      */
      FreeNode(agx::Real worldX, agx::Real worldY, agx::Real worldZ);

      DOXYGEN_START_INTERNAL_BLOCK()
    public:
      FreeNode(agx::RigidBody* body, const agx::Vec3& halfExtents, agx::Constraint* constraint);
      AGXSTREAM_DECLARE_SERIALIZABLE_CUSTOM_CREATE(agxCable::FreeNode);

    protected:
      FreeNode();
      virtual ~FreeNode();

      DOXYGEN_END_INTERNAL_BLOCK()
  };
}

#endif
