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

#include <agxSDK/LinkedSegment.h>

#include <agx/Hinge.h>

namespace agx
{
  class MergedBody;
}

namespace agxVehicle
{
  class Track;
  AGX_DECLARE_POINTER_TYPES( TrackNode );

  using IndexRange          = agx::IndexRangeT<agx::UInt>;
  using IndexRangeContainer = agx::VectorPOD<IndexRange>;

  /**
  Node/segment/shoe object in agxVehicle::Track. A track node has well defined
  extents that can be independent of the actual shape of it. The node is constrained
  with the node before in the track list.

  The model center is by definition the anchor position of the constraint with
  constraint/hinge axis pointing in the y direction of the body.
  */
  class AGXVEHICLE_EXPORT TrackNode : public agxSDK::LinkedSegment
  {
    public:
      /**
      Construct given transform and half extents of the shape. Note that a
      rigid body instance without a geometry/shape is created.
      \param transform - initial transform of this node
      \param halfExtents - half extents of this node, defining length, width
                           and thickness of the track
      */
      TrackNode( const agx::AffineMatrix4x4& transform, const agx::Vec3& halfExtents );

      /**
      \return the width of this node
      */
      agx::Real getWidth() const;

      /**
      \return the thickness of this node
      */
      agx::Real getThickness() const;

      /**
      \return the rotation axis in world coordinate frame
      */
      agx::Vec3 getRotationAxis() const;

      /**
      \return merged body if this node is merged
      */
      agx::MergedBody* getMergedBody() const;

      /**
      \return true if this node is merged with a wheel - otherwise false
      */
      agx::Bool isMergedWithWheel() const;

      /**
      \return the track this node is part of - null if not added to a track
      */
      agxVehicle::Track* getTrack() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxVehicle::TrackNode );

    protected:
      /**
      Default constructor used by serialization.
      */
      TrackNode();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~TrackNode();
  };
}
